# 🤟 ASL Recognition

A comprehensive benchmark of **seven deep learning architectures** for static American Sign Language (ASL) hand sign recognition. This project compares 3D‑CNN, CNN‑LSTM, CNN‑BiLSTM, Attention CNN‑BiLSTM, TCN, Vision Transformer (ViT), and a custom Two‑Stream Static network – all implemented in PyTorch and evaluated on a consistent dataset.

> **Target audience**: Researchers and practitioners in computer vision, sign language recognition, and embedded AI systems who need an evidence‑based comparison of temporal vs. spatial architectures on a purely static image task.

---

## ✨ Key Features

### 🧠 Model Zoo (7 Architectures)
| Model | Type | Parameters |
|-------|------|------------|
| 3D‑CNN | Volumetric (video) | 2.38 M |
| CNN‑LSTM | Spatial + sequential | 8.75 M |
| CNN‑BiLSTM | Bidirectional sequential | 17.4 M |
| Attention CNN‑BiLSTM | BiLSTM + attention | 17.5 M |
| TCN | Dilated causal convolutions | 3.09 M |
| Vision Transformer (ViT) | Transformer on image patches | 3.24 M |
| Two‑Stream Static | RGB + edge streams | 0.75 M (2.02 M after refactor) |

### 📊 Quantitative Benchmarking
- **Accuracy & loss curves** over 50 epochs (all models)
- **Parameter efficiency** (accuracy per million parameters)
- **Weight statistics** (mean, std, min/max, initialisation sensitivity)
- **Computational cost** (FLOPs estimation)
- **Overfitting analysis** (train‑test gap)

### 🎯 Real‑Time Hand Detection (Optional)
- YOLOv11‑based hand segmentation (`image_segmentation.py`)
- Crops and resizes hand region to 64×64 before feeding into any model

### 📁 Model Parameter Analysis
- Automated extraction of all layer‑wise parameters (`parameters.py`)
- Summary tables and distribution histograms (saved in `Result/model_parameters/`)

---

## 🧩 Full‑Stack Architecture

| Layer               | Technology |
|---------------------|------------|
| Deep Learning       | PyTorch, torchvision |
| Model Definitions   | `models.py` (7 classes) |
| Training Scripts    | Separate `.py` for each model |
| Hand Detection      | Ultralytics YOLO (yolo11n.pt) |
| Data Handling       | Custom `SequenceImageDataset`, PIL, torch DataLoader |
| Visualisation       | Matplotlib (loss/accuracy curves) |
| Parameter Analysis  | Custom script (mean/std/min/max per layer) |

---

## 🚀 Getting Started

### Prerequisites
- **Python** 3.10+
- **pip** + virtual environment (recommended)
- **PyTorch** (CPU or CUDA)
- Optional: **Ultralytics** for YOLO hand detection

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ASL-Recognition.git
cd ASL-Recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision matplotlib pillow ultralytics
```

### Dataset Preparation

Place your image sequences in the following structure:

```
train_data/
├── class_A/
│   ├── seq1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── seq2/
│       └── ...
└── class_B/
    └── ...
```

Each sequence folder contains 10 frames (padded/truncated automatically).  
Images are resized to 64×64 (or 32×32 for TCN memory optimisation).

### Training a Model

```bash
# Example: train CNN‑LSTM
python Code/cnn_lstm.py

# Other models
python Code/cnn_bi_lstm.py
python Code/cnn_bi_lstm_attention.py
python Code/3d_cnn.py
python Code/tcn.py
python Code/two_stream_network.py
python Code/vision_image_transformer.py
```

All training scripts save weights to `pre trained weigths/` and display loss/accuracy curves at the end.

### Parameter Analysis (after training)

```bash
python Code/parameters.py
```

This generates detailed parameter reports (per‑layer mean/std, total counts, distribution) in `Result/model_parameters/`.

### Real‑Time Hand Detection (Optional)

```bash
python Code/image_segmentation.py
```

Press **s** to save the current cropped hand image (saved to `test_dataset/4/`), **q** to quit.

---

## 📡 Model Inference (Example)

```python
from models import CNNLSTM
import torch

model = CNNLSTM(num_classes=26)
model.load_state_dict(torch.load('pre trained weigths/cnn_lstm_weights.pth'))
model.eval()

# frames: tensor of shape (1, 10, 3, 64, 64)
with torch.no_grad():
    logits = model(frames)
    pred = torch.argmax(logits, dim=1)
```

---

## 📁 Project Structure

```
ASL_RECOGNITION/
├── README.md                                 # This file
├── ASL RECOGNITION (3rd Semester).pdf       # Full project report
├── PROJECT REPORT.docx                       # Detailed methodology
├── weights.docx                              # Extracted weight statistics
│
├── Code/
│   ├── models.py                             # All 7 model definitions
│   ├── cnn_lstm.py                           # CNN‑LSTM training
│   ├── cnn_bi_lstm.py                        # CNN‑BiLSTM training
│   ├── cnn_bi_lstm_attention.py              # Attention variant
│   ├── 3d_cnn.py                             # 3D CNN training
│   ├── tcn.py                                # TCN training
│   ├── two_stream_network.py                 # Two‑Stream Static training
│   ├── vision_image_transformer.py           # ViT training
│   ├── parameters.py                         # Parameter analysis utility
│   ├── image_segmentation.py                 # YOLO hand detection
│   └── yolo11n.pt                            # Pre‑trained YOLO weights
│
└── Result/
    ├── model_parameters/                     # Auto‑generated param reports
    │   ├── *_parameters.txt                  # Per‑layer mean/std/min/max
    │   └── *_summary.txt                     # Total params & distribution
    └── *.png                                 # Loss/accuracy curves (50 epochs)
```

---

## 🔬 Methodology (Academic Foundations)

| Module | Method | Reference |
|--------|--------|-----------|
| **Spatial feature extraction** | 2D/3D convolutions, patch embedding | LeCun et al. (1998); Dosovitskiy et al. (2020) |
| **Temporal modelling** | LSTM, BiLSTM, TCN (dilated causal conv) | Hochreiter & Schmidhuber (1997); Bai et al. (2018) |
| **Attention** | Additive (Bahdanau) attention over BiLSTM states | Bahdanau et al. (2014) |
| **Two‑Stream** | RGB + edge gradient streams | Simonyan & Zisserman (2014) – adapted to static |
| **Evaluation metrics** | Accuracy, parameter count, training stability | – |

### Key Research Questions Addressed

1. **Can temporal architectures (LSTM, TCN) perform well on purely spatial tasks?**  
   → Yes: TCN achieved **90%** test accuracy, outperforming most models.

2. **Is parameter count the main driver of performance?**  
   → No: Two‑Stream (0.75 M) outperforms BiLSTM (17.4 M) with much fewer parameters.

3. **Do Vision Transformers work on small, specialised datasets?**  
   → No (without massive pre‑training): ViT collapsed to <5% accuracy.

4. **What weight initialisation issues cause training failure?**  
   → ViT’s `cls_token` and `pos_embedding` had extreme values (±3 σ), destabilising training.

---

## 📊 Key Results

### Performance Ranking (Test Accuracy)

| Rank | Model | Accuracy | Parameters | Efficiency (params/acc%) |
|------|-------|----------|------------|--------------------------|
| 1 | **TCN** | **90%** | 3.09 M | 38,615 |
| 2 | Two‑Stream Static | 80% | 0.75 M | **10,051** |
| 3 | CNN‑BiLSTM‑Attention | 70% | 17.4 M | 248,753 |
| 4 | CNN‑BiLSTM | 65% | 17.4 M | 267,880 |
| 5 | CNN‑LSTM | 60% | 8.75 M | 145,879 |
| 6 | ViT | <5% | 3.24 M | 808,031 |
| 7 | 3D‑CNN | <10% | 2.38 M | 297,923 |

### Training Stability Observations

| Model | Weight Std Dev | Gradient Flow | Overfitting |
|-------|----------------|---------------|-------------|
| TCN | 0.01–0.05 | Very stable | Moderate |
| Two‑Stream (edge stream) | 0.52 (high) | Unstable early | Severe (spatial only) |
| ViT | 0.02–0.05 (but extreme embeddings) | Complex, slow | Underfitting |
| CNN‑BiLSTM | 0.04–0.16 | Stable | Low |

> Full weight statistics (mean, std, min, max per layer) are available in `Result/model_parameters/*_parameters.txt`.

---

## ⚠️ Known Limitations & Model Risks

1. **Small dataset size** – The training set is limited (a few thousand clips), causing overfitting in 3D‑CNN and Two‑Stream.
2. **No pretrained weights** – ViT and 3D‑CNN were trained from scratch; they would benefit from ImageNet or Kinetics pretraining.
3. **TCN spatial blindness** – Our TCN implementation flattens each frame to a vector, losing spatial structure. A CNN front‑end would improve performance.
4. **Two‑Stream edge map** – Simple gradient magnitude is crude; optical flow would provide richer motion cues (but requires multiple frames).
5. **Fixed sequence length** – All models use 10 frames (padding/truncation). Longer or variable‑length sequences may improve LSTM‑based models.
6. **No cross‑validation** – Performance may vary with different train/test splits.

> See the full project report (PDF) for a detailed discussion and a roadmap for addressing these limitations.

---

## 🔮 Future Roadmap

| Timeframe | Enhancements |
|-----------|---------------|
| **Near‑term (0–3 mo)** | Add pretrained backbones (ResNet18 for CNN‑LSTM, ViT‑pretrained for transformer); implement spatial front‑end for TCN; apply stronger regularisation (Dropout, weight decay). |
| **Medium‑term (3–6 mo)** | Expand dataset (more signers, lighting conditions, backgrounds); implement cross‑validation; deploy the best model (TCN or CNN‑LSTM) as a real‑time web app with FastAPI + Next.js. |
| **Long‑term (6–12 mo)** | Incorporate dynamic sign recognition (video‑based) using the best temporal models; explore ensemble methods (TCN + Two‑Stream); publish benchmark results. |

---
