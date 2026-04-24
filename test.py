

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHECKPOINT_PATH = r"C:\dlinknet_local_epoch_70.pth" 
IMAGE_PATH = r"C:\Users\amolk\Downloads\data_0040.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. D-LINKNET ARCHITECTURE
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
        resnet = models.resnet34(weights=None)
        
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
# 3. INITIALIZATION
# ==========================================
print("Loading model for diagnostics...")
model = DLinkNet34(num_classes=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==========================================
# 4. SINGLE IMAGE INFERENCE
# ==========================================
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Could not find image at {IMAGE_PATH}")
    exit()

print("Analyzing image...")
original_img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
orig_h, orig_w = original_img.shape[:2]

augmented = transform(image=original_img)
image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        output = model(image_tensor)
        probability_map = torch.sigmoid(output)
        
        # Extract raw probabilities BEFORE thresholding for the heatmap
        raw_probs = probability_map.squeeze().cpu().numpy()
        
        # Standard threshold for drawing the green boxes
        prediction = (probability_map > 0.5).float().squeeze().cpu().numpy()

# Resize both back to the original image dimensions
raw_probs_resized = cv2.resize(raw_probs, (orig_w, orig_h))
pred_resized = cv2.resize(prediction, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# ==========================================
# 5. DRAW GREEN BOUNDING BOXES
# ==========================================
pred_uint8 = (pred_resized * 255).astype(np.uint8)
contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxed_img = original_img.copy()
roof_count = 0

for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        roof_count += 1

print(f"✅ Hard Detection Count: {roof_count} roofs found.")

# ==========================================
# 6. DISPLAY HEATMAP DIAGNOSTICS
# ==========================================
print("Popping up diagnostic window...")

plt.figure(figsize=(18, 6))

# Original
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title("Original Drone Image")
plt.axis('off')

# AI Heatmap (Blue = 0%, Red = 100% Confident)
plt.subplot(1, 3, 2)
img_plot = plt.imshow(raw_probs_resized, cmap='jet')
plt.title("AI Confidence Heatmap")
plt.colorbar(img_plot, fraction=0.046, pad=0.04)
plt.axis('off')

# Green Box Overlay
plt.subplot(1, 3, 3)
plt.imshow(boxed_img)
plt.title(f"Strict Detections (>50% Confidence)")
plt.axis('off')

plt.tight_layout()
plt.show()
