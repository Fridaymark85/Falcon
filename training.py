import os
import warnings
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch.amp as amp
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 0. SILENCE ALL WARNINGS
# ==========================================
# Suppresses PIL warnings regarding unknown GeoTIFF tags
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ==========================================
# 1. LOCAL CONFIGURATION & CLEANING
# ==========================================
DATASET_DIR = r"C:\Users\amolk\Downloads\kollaru_roof_cleaned" 
SAVE_DIR = r"C:\training_checkpoints_kollaru"

IMG_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 2             # RTX 3050 6GB Protection
ACCUMULATION_STEPS = 4     # Simulates Batch Size 8
START_EPOCH = 0 
TOTAL_EPOCHS = 80 
LEARNING_RATE = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"System Check: Training on {DEVICE.upper()} | Ryzen CPU handling Dataloader")

# --- THE BLACK TILE PURGE (PIL VERSION) ---
def clean_dataset(image_dir, mask_dir, threshold=0.90):
    if not os.path.exists(image_dir): 
        print(f"Directory missing: {image_dir}")
        return
        
    all_images = os.listdir(image_dir)
    removed_count = 0
    print(f"\nScanning {len(all_images)} images to purge useless black tiles...")
    
    for img_name in tqdm(all_images, desc="Cleaning Data"):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        try:
            # Load in grayscale to detect empty/black zones
            img_gray = np.array(Image.open(img_path).convert("L"))
            black_ratio = np.sum(img_gray == 0) / img_gray.size
            
            if black_ratio >= threshold:
                os.remove(img_path)
                if os.path.exists(mask_path): 
                    os.remove(mask_path)
                removed_count += 1
        except Exception:
            continue
            
    print(f"✅ Purged {removed_count} useless tiles. {len(os.listdir(image_dir))} valid images remain.\n")

clean_dataset(IMG_DIR, MASK_DIR, threshold=0.90)

# ==========================================
# 2. D-LINKNET ARCHITECTURE (ROAD NETWORK FOCUS)
# ==========================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu3(self.norm3(self.conv3(self.relu2(self.norm2(self.deconv2(self.relu1(self.norm1(self.conv1(x)))))))))

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        # Dilated convolutions expand the model's receptive field to connect broken road networks
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d4 = self.relu(self.dilate4(d2))
        d8 = self.relu(self.dilate8(d4))
        return x + d1 + d2 + d4 + d8 

class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(DLinkNet34, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)
        
        self.finaldeconv1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        e0 = self.firstmaxpool(self.firstrelu(self.firstbn(self.firstconv(x))))
        
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        e4 = self.dblock(e4)
        
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return self.finalconv3(self.finalrelu2(self.finalconv2(self.finalrelu1(self.finaldeconv1(d1)))))

# ==========================================
# 3. DATASET & HEAVY AUGMENTATIONS
# ==========================================
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # Shadow and Drone Imagery Specific Augmentations
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4), 
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    
    # Structural shifts
    A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"❌ ERROR: Missing folders! Make sure {image_dir} and {mask_dir} exist.")
            self.valid_images = []
            return

        self.valid_images = [img for img in os.listdir(image_dir) if os.path.exists(os.path.join(mask_dir, img))]
        print(f"✅ Loaded {len(self.valid_images)} valid images for training.")

    def __len__(self): return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.valid_images[idx])
        mask_path = os.path.join(self.mask_dir, self.valid_images[idx])

        # PIL silent loading to bypass GeoTIFF C++ warnings
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.unsqueeze(0)

dataset = RoadDataset(IMG_DIR, MASK_DIR, transform=train_transform)

if len(dataset) == 0:
    print("Halting script because no data was found.")
    exit()

# num_workers=0 is MANDATORY for Windows local execution
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# ==========================================
# 4. ARCHITECTURE, TOPOLOGICAL LOSS & SCHEDULER
# ==========================================
model = DLinkNet34(num_classes=1).to(DEVICE)

# Focal Loss targets hard-to-see dirt paths. Tversky heavily punishes breaking the road network.
focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
tversky_loss_fn = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)

def topological_loss(predictions, masks):
    return focal_loss_fn(predictions, masks) + tversky_loss_fn(predictions, masks)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
scaler = amp.GradScaler('cuda')

def calculate_metrics(predictions, masks):
    probs = torch.sigmoid(predictions)
    preds = (probs > 0.5).float()
    
    correct = (preds == masks).sum().item()
    total = torch.numel(masks)
    accuracy = correct / total
    
    intersection = (preds * masks).sum().item()
    union = preds.sum().item() + masks.sum().item() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return accuracy, iou

# ==========================================
# 5. OOM-SAFE TRAINING LOOP WITH TQDM
# ==========================================
print(f"\n🚀 Starting Local D-LinkNet Topological Training on C:\\ Drive!")

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
    model.train()
    epoch_loss, epoch_acc, epoch_iou = 0, 0, 0
    valid_batches = 0

    current_lr = optimizer.param_groups[0]['lr']
    
    # Clean TQDM Loop
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [LR: {current_lr:.1e}]", leave=True)
    
    optimizer.zero_grad() 
    
    for i, (images, masks) in enumerate(loop):
        try:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            with amp.autocast('cuda'):
                predictions = model(images)
                loss = topological_loss(predictions, masks) 
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_acc, batch_iou = calculate_metrics(predictions, masks)
            
            epoch_loss += (loss.item() * ACCUMULATION_STEPS) 
            epoch_acc += batch_acc
            epoch_iou += batch_iou
            valid_batches += 1
            
            loop.set_postfix(loss=(loss.item() * ACCUMULATION_STEPS), acc=batch_acc, iou=batch_iou)

        except torch.cuda.OutOfMemoryError:
            # Flushes memory if the RTX 3050 6GB hits the limit
            print(f"\n⚠️ WARNING: GPU OOM caught at batch {i}. Recovering and skipping...")
            torch.cuda.empty_cache()  
            gc.collect()              
            optimizer.zero_grad()     
            continue                  

    if valid_batches > 0:
        avg_loss = epoch_loss / valid_batches
        avg_acc = epoch_acc / valid_batches
        avg_iou = epoch_iou / valid_batches
        
        scheduler.step(avg_loss)

        print(f"\nEpoch {epoch+1} Summary: Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | IoU: {avg_iou:.4f}\n")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == TOTAL_EPOCHS:
            save_filename = f"dlinknet_local_epoch_{epoch+1}.pth"
            save_path = os.path.join(SAVE_DIR, save_filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
                'iou': avg_iou
            }, save_path)
            print(f"💾 Checkpoint saved: {save_filename}")
    else:
        print("\n❌ CRITICAL: Entire epoch failed due to OOM. Drop BATCH_SIZE to 1.")

print("Training Complete! The network is ready.")