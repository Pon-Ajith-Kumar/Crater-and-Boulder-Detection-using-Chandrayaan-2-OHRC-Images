#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ### Dataset and DataLoader

# In[2]:


class OHRCDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for class_name in sorted(os.listdir(root_dir)):
            img_dir = os.path.join(root_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)
            if os.path.isdir(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(img_dir, img_name))
                        self.mask_paths.append(os.path.join(mask_class_dir, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


# In[3]:


def load_data(train_dir, val_dir, test_dir, batch_size):
    trainset = OHRCDataset(train_dir+"/images", train_dir+"/masks", transform)
    valset = OHRCDataset(val_dir+"/images", val_dir+"/masks", transform)
    testset = OHRCDataset(test_dir+"/images", test_dir+"/masks", transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ### Evaluation Metrics

# In[4]:


def threshold(pred, thresh=0.5):
    """Convert probability map to binary mask."""
    return (torch.sigmoid(pred) > thresh).float()

def dice_score(pred, target, smooth=1e-6):
    pred = threshold(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + smooth) / (union + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = threshold(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    pred = threshold(pred)
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total

# --- Combined BCE + Dice Loss ---
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - ((2 * inter + smooth) / (union + smooth))

def combined_loss(pred, target):
    return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)


# ## U-Net Model

# In[5]:


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(n_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = conv_block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = self.pool(x3)
        x5 = self.enc3(x4)
        x6 = self.pool(x5)
        x7 = self.enc4(x6)
        x8 = self.pool(x7)
        x9 = self.bottleneck(x8)

        x = self.up1(x9)
        x = self.dec1(torch.cat([x, x7], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x5], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up4(x)
        x = self.dec4(torch.cat([x, x1], dim=1))
        return self.out_conv(x)


# ## Training with metrics per epoch

# In[6]:


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer, scheduler, and scaler
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(device="cuda")

    best_val_loss = np.inf
    patience_counter = 0

    print("Training Started...\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_stats = {"dice": 0, "iou": 0, "acc": 0}

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # -----------------------------
        # Training Phase
        # -----------------------------
        for img, mask in loop:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()

            # Mixed Precision forward pass
            with autocast(device_type="cuda"):
                pred = model(img)
                loss = combined_loss(pred, mask)  # BCE + Dice loss

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()

            # Accumulate metrics
            train_stats["dice"] += dice_score(pred, mask).item()
            train_stats["iou"] += iou_score(pred, mask).item()
            train_stats["acc"] += pixel_accuracy(pred, mask).item()

        # Normalize train stats
        for k in train_stats:
            train_stats[k] /= len(train_loader)
        train_loss /= len(train_loader)

        # -----------------------------
        # Validation Phase
        # -----------------------------
        model.eval()
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                with autocast(device_type="cuda"):
                    pred = model(img)
                    val_loss += combined_loss(pred, mask).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Learning rate log (optional)
        if scheduler.num_bad_epochs == 0:
            print(f"LR reduced to {scheduler.optimizer.param_groups[0]['lr']:.2e}")

        # -----------------------------
        # Epoch Summary
        # -----------------------------
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice: {train_stats['dice']:.3f}, IoU: {train_stats['iou']:.3f}, Acc: {train_stats['acc']:.3f}")

        # -----------------------------
        # Early Stopping
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")


# ## Test Evaluation Helper Function

# In[12]:


def test_evaluation(model, loader):
    model.eval()
    model.to(device)
    stats = {"Dice":0, "IoU":0, "Accuracy":0}

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            stats["Dice"] += dice_score(pred, mask)
            stats["IoU"] += iou_score(pred, mask)
            stats["Accuracy"] += pixel_accuracy(pred, mask)

    for k in stats: stats[k] /= len(loader)
    df = pd.DataFrame([stats])
    print("\nTEST METRICS SUMMARY")
    display(df.style.set_caption("Segmentation Performance Metrics").format("{:.4f}"))
    return df


# ## Visualize predictions and ground truth

# In[8]:


def show_all_visualizations(model, loader, num_samples=3):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(imgs))
        preds = (preds > 0.5).float()

    for i in range(min(num_samples, len(imgs))):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        mask_gt = masks[i][0].cpu().numpy()
        mask_pred = preds[i][0].cpu().numpy()

        if img.max() > 1:
            img = img / 255.0

        # --- Overlay (Red=Prediction, Green=GT) ---
        overlay = img.copy()
        overlay[..., 0] += mask_pred * 0.8
        overlay[..., 1] += mask_gt * 0.8
        overlay = np.clip(overlay, 0, 1)

        # --- Contour visualization ---
        image_cv = (img * 255).astype(np.uint8)
        pred_cv = (mask_pred * 255).astype(np.uint8)
        vis = image_cv.copy()

        contours, _ = cv2.findContours(pred_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per == 0: continue
            circ = 4 * np.pi * area / (per ** 2)
            if area > 20:
                cv2.drawContours(vis, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(vis, f"{circ:.2f}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        gt_cv = (mask_gt * 255).astype(np.uint8)
        contours_gt, _ = cv2.findContours(gt_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_gt, -1, (0,255,0), 1)  # Green = GT

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        # --- Combined 2x2 Layout ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Input Image"); axes[0, 0].axis("off")

        axes[0, 1].imshow(mask_gt, cmap="gray")
        axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")

        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title("Overlay (Red=Pred, Green=GT)"); axes[1, 0].axis("off")

        axes[1, 1].imshow(vis_rgb)
        axes[1, 1].set_title("Contours (Red=Pred, Green=GT, Label=Circularity)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()


# ## Training UNet Model

# In[9]:


train_loader, val_loader, test_loader = load_data(
    "/teamspace/uploads/OHRC-ISRO/dataset/train",
    "/teamspace/uploads/OHRC-ISRO/dataset/val",
    "/teamspace/uploads/OHRC-ISRO/dataset/test",
    batch_size=4
)

model = UNet()
train_model(model, train_loader, val_loader, num_epochs=100)


# ## Evaluation Result

# In[13]:


# Evaluate on test or validation loader
test_results = test_evaluation(model, val_loader)


# ## Visualization

# In[15]:


get_ipython().system('pip install opencv-python')
import cv2


# In[16]:


# Show sample predictions
show_all_visualizations(model, val_loader, num_samples=1)


# In[ ]:




