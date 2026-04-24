# Project Falcon: AI Diagnostic Tool for Swamitva Scheme

## 📖 Overview
**Project Falcon** is an advanced AI system designed to automate the analysis of drone imagery for the **Swamitva Scheme**. It utilizes a custom **D-LinkNet34** architecture to perform semantic segmentation on aerial photographs, specifically targeting:
- **Roof Type Detection:** Identifying Clay Tile, Concrete, and Tin roofs.
- **Road Network Mapping:** Extracting road infrastructure from drone footage.

The system is optimized for local execution on consumer-grade hardware (specifically NVIDIA RTX 3050 with 4GB-6GB VRAM) and includes robust memory management to prevent Out-Of-Memory (OOM) crashes during training.

---

## 🚀 Key Features

### 1. Advanced Architecture (D-LinkNet34)
- Based on **ResNet-34** with a specialized **D-block** (Dilated Convolution Block) to expand the receptive field.
- **Decoder Blocks:** Utilizes transposed convolutions for precise upscaling and boundary detection.
- **Topological Loss:** Combines **Focal Loss** (for hard-to-see targets) and **Tversky Loss** (to prevent broken road networks).

### 2. Hardware Optimization
- **Gradient Accumulation:** Simulates larger batch sizes on limited VRAM by accumulating gradients over multiple steps.
- **Mixed Precision Training:** Uses automatic mixed precision for faster training and reduced memory usage.
- **OOM Recovery:** Automatic catch-and-retry mechanism if the GPU hits memory limits during training.

### 3. Data Intelligence
- **Automated Data Cleaning:** Script to detect and remove useless black/empty tiles from the dataset before training.
- **Drone-Specific Augmentations:** 
  - Random shadows, CLAHE (Contrast Limited Adaptive Histogram Equalization), and elastic transforms to mimic real-world drone conditions.

### 4. Diagnostic Inference
- **Heatmap Visualization:** Generates confidence heatmaps to visualize model certainty.
- **Bounding Box Overlay:** Draws detection boxes on structures with a configurable confidence threshold.
- **Counting Logic:** Automatically counts detected roofs/structures.

---

## 📂 Project Structure

```text
Project_Falcon/
│
├── test.py                  # Inference & Diagnostics (Heatmap & Counting)
├── training.py              # Training loop with OOM protection & Augmentations
├── .gitattributes           # Git LFS configuration for binary files
├── .gitignore               # Ignored files (logs, cache, etc.)
│
└── trained_weights/         # Pre-trained model checkpoints (Git LFS managed)
    ├── trained_weights_CC_roof.pth      # Clay/Concrete Roof weights
    ├── trained_weights_Kollaru_roof.pth # Kollaru region specific weights
    ├── trained_weights_roadNetwork.pth  # Road network weights
    └── trained_weights_tin_roof.pth     # Tin Roof weights
