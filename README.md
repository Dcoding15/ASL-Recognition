# 🤟 ASL Recognition

## 1. Quantitative Comparison of Model Complexity

### Total Parameter Counts and Largest Component

| Model                     | Total Parameters | Largest Component (% of total)        |
|---------------------------|------------------|----------------------------------------|
| CNN3D                     | 2.38 M           | 3D Conv layers (~95%)                  |
| CNNLSTM                   | 8.75 M           | LSTM (64.6%), then FC (35.4%)          |
| CNNBiLSTM                 | 17.4 M           | BiLSTM (79.3%), then FC (20.7%)        |
| AttentionCNNBiLSTM        | 17.5 M           | BiLSTM (78.9%), Attention (negligible) |
| TCN                       | 3.09 M           | Temporal Conv blocks (~90%)            |
| TwoStreamStatic           | 2.02 M           | Spatial stream CNN (~60%), Temporal CNN (~35%) |
| ViT (from scratch)        | 3.24 M           | Transformer encoder (6 layers, 8 heads, dim 256) – ~70% |

*Note: CNNLSTM’s LSTM hidden dimension is likely 256; BiLSTM doubles it (2×256 each direction).*

### Approximate FLOPs and Relative Computational Cost
- **3D CNN**: ~5.8 GFLOPs per 10‑frame clip (3×3×3 kernels, four 3D conv layers). Very high cost due to volumetric convolutions.
- **CNN‑LSTM / CNN‑BiLSTM**: ~0.9 GFLOPs per clip (2D CNN applied frame‑wise, then LSTM over 10 time steps). CNN front‑end is efficient; LSTM adds sequential cost.
- **TCN**: ~1.1 GFLOPs (1D dilated convolutions on 10‑step sequences of 128‑dim features). Much cheaper than 3D CNN.
- **TwoStreamStatic**: ~0.3 GFLOPs per single image (two parallel 2D CNNs). It ignores temporal context but is extremely fast.
- **ViT**: ~2.1 GFLOPs (patch embedding + 6 transformer layers on 10 frames averaged later). Transformer self‑attention is quadratic in sequence length, but here frames are processed independently; still moderate cost.

**Parameter efficiency**: CNN3D is small (2.38 M) but overfits dramatically. TCN (3.09 M) and ViT (3.24 M) offer compact representations yet struggle to learn the task. The CNNLSTM (8.75 M) is the most balanced in terms of performance vs. size. CNNBiLSTM and its attention variant are heavily over‑parameterized for this dataset (17.4–17.5 M). TwoStreamStatic (2.02 M) is the smallest but fundamentally limited by the absence of temporal modelling.

---

## 2. Analysis of Overfitting & Generalization (from loss/accuracy curves)

All curves were visually analysed over 50 epochs. A summary of their behaviour:

| Model                   | Train Loss | Test Loss      | Train Acc. | Test Acc.   | Overfitting Severity |
|-------------------------|------------|----------------|------------|-------------|----------------------|
| CNN3D                   | ↓ steeply  | ↗ after ~10    | → ~99%     | ~40% peak   | Severe               |
| CNNLSTM                 | ↓ steadily  | ↓ then plateaus | → ~90%     | ~52% steady | Mild underfit        |
| CNNBiLSTM               | ↓ smoothly  | ↓ slowly        | → ~88%     | ~55% peak   | Low                  |
| AttentionCNNBiLSTM      | ↓ more slowly | ↓ gradually  | → ~85%     | ~54% peak   | Low                  |
| TCN                     | ↓ steadily  | ↗ after ~20    | → ~92%     | ~48% peak   | Moderate             |
| TwoStreamStatic         | ↓ quickly   | ↗ after ~5     | → ~80%     | ~65% plateau| Severe (spatial only)|
| ViT                     | ↓ very slowly| ↓ barely      | → ~45%     | ~33% after 50| Underfit (slow)      |

**Key observations:**
- **CNN3D**: Test loss starts increasing around epoch 10, while train accuracy saturates near 100%. The large gap signals a severe lack of regularisation for the 3D conv kernels; the model memorises training videos but fails to generalise.
- **CNNLSTM**: Both curves converge quickly but train accuracy plateaus below 95%, test accuracy stays around 52% – suggesting the model could benefit from more capacity or a longer training schedule (slight underfitting).
- **CNNBiLSTM & Attention variant**: More stable, test loss continues to decrease slowly, test accuracy peaks around epoch 35–40. The bidirectional LSTM captures temporal context better, but the high parameter count is not fully exploited.
- **TCN**: Overfitting gap appears after epoch 20; test loss rises while train loss keeps falling. The dilated convolutions need stronger regularisation (e.g., dropout in residual blocks).
- **TwoStreamStatic**: The test accuracy plateaus near 65% very early (epoch 5–10), and test loss climbs afterwards because the model can only learn from static appearance and edge maps; it cannot encode motion.
- **ViT**: Extremely slow convergence; test loss barely drops, final test accuracy around 30–35%. Training from scratch on a small dataset with full‑size ViT is notoriously difficult; lack of inductive biases hurts.

---

## 3. Weight Statistics & Training Dynamics

### CNN3D (from `3d_cnn_weights_parameters.txt`)
- **First 3D conv layer**: mean = −0.0065, std = 0.064. Values are close to zero, which is typical early in training but may indicate vanishing gradients if std remains small after 50 epochs. Later layers show even smaller mean and std (e.g., 0.001 mean, 0.01 std). This often happens when ReLU units saturate or when the network learns trivial features; it can explain overfitting because the model may rely on a few high‑magnitude weights, leaving many filters under‑utilised.
- **FC layers**: Means near zero, std ~0.1–0.2, min/max in ±0.5 range. No extreme values, suggesting no gradient explosion.

### ViT (from `vit_weights_parameters.txt`)
- **cls_token**: mean ≈ 0.0, std ≈ 1.02. The token’s large variance (∼1) indicates it carries significant information and is updated strongly – typical for a learnable classification token.
- **pos_embedding**: mean ≈ 0.0, std ≈ 1.05. The high std shows the position embeddings are actively learning, but their near‑zero mean is expected.
- **Transformer block weights**: Q/K/V matrices and MLP weights have mean ~0.0, std ~0.02–0.05. Such small standard deviations point to extremely cautious updates; the network has not yet extracted meaningful features. This aligns with the very low test accuracy – the ViT is essentially still near random initialisation after 50 epochs.
- **LayerNorm weights**: all near 1.0, bias near 0.0 – normal for LayerNorm, shows proper initialisation.

### Two‑Stream Static (temporal edge stream)
- **temporal CNN (edge input)**: conv layer weights have std up to 0.52, min = −1.30, max = +1.32. These are considerably larger than spatial stream weights (std ~0.1). The edge maps (hand‑crafted) produce strong gradients, causing the temporal stream to dominate early training. Such large magnitudes can lead to instability; the edge stream might be learning noisy, high‑frequency patterns that don’t generalise.

### LSTM‑based vs. TCN
- **LSTM models**: Input‑to‑hidden and hidden‑to‑hidden weight matrices show means around 0.0, std 0.02–0.05. Gate biases are usually close to 0 (forget gate bias ≈ 1 after initialisation, but after training they might drift slightly). These small values are characteristic of well‑regularised recurrent models.
- **TCN**: 1D conv kernel weights have means near 0.0 and std 0.04–0.08, slightly higher than LSTM matrices. No anomalous extremes, but the overfitting indicates the network relies too much on spurious temporal correlations.

---

## 4. Architectural Strengths & Weaknesses

### 3D CNN
- **Why overfitting?** The network has no Dropout in 3D conv layers, no weight decay, and the dataset is likely small (a few thousand clips). Spatiotemporal filters can easily memorise clip‑specific motion patterns. The frame size (64×64) and 10 frames give a large input volume, but the training set is insufficient.
- **Mitigation**: Add spatial‑temporal Dropout (e.g., Dropout3d with p=0.3‑0.5), use strong L2 regularisation, apply data augmentation (random cropping, time warping, horizontal flipping). Consider Kinetics‑pretrained 3D ResNets if transfer learning is allowed.

### CNN‑LSTM vs. CNN‑BiLSTM
- The unidirectional LSTM (8.75 M) achieved about 52% test accuracy; the bidirectional one (17.4 M) reached ~55%. The 3% gain does not justify doubling parameters. The BiLSTM probably over‑smooths the short 10‑frame sequence, and the extra backward pass provides diminishing returns. **CNNLSTM is more parameter‑efficient**.

### AttentionCNNBiLSTM
- The attention mechanism adds negligible parameters (a small two‑layer MLP). Loss curves show slightly slower convergence than BiLSTM but similar final accuracy. For 10‑frame sequences, the model can easily attend to all frames without selective focus. Attention becomes more beneficial when sequences are longer (e.g., 30+ frames) and some frames are irrelevant. Here, its impact is marginal.

### TCN
- With dilated convolutions and residual connections, the TCN has a larger effective receptive field (covering the whole sequence) than a single‑layer LSTM. It can capture long‑range dependencies efficiently and is parallelisable. The moderate overfitting suggests it is a promising candidate if regularised properly. TCN’s small size (3.09 M) and good temporal modelling make it suitable for deployment on edge devices.

### Two‑Stream Static
- **Ignoring temporal information is catastrophic** for sign language, where motion trajectory, hand shape change, and timing are essential. A single image cannot distinguish many signs (e.g., static letters ‘V’ vs. ‘U’ look similar). Even with an edge map, the model plateaus at ~65%, far below sequence models.
- **Conversion to sequential two‑stream**: Process multiple frames with a shared CNN, then pool or use an LSTM over time. Alternatively, compute optical flow between consecutive frames as the temporal stream input.

### Vision Transformer (ViT)
- **Why it struggles**: ViTs lack the built‑in translation equivariance and locality of CNNs. They require massive data or strong augmentation to learn spatial structures. The model has 6 transformer layers (256 dim, 8 heads) – a relatively large model for a small dataset. Training from scratch with only 50 epochs and no pretraining leads to poor convergence.
- **Improvements**: Reduce to 2–3 layers, use smaller patch size (e.g., 4×4 instead of 8×8), incorporate convolutional patch embedding (hybrid ViT), apply heavy data augmentation (RandAugment, MixUp), and importantly use a pretrained ViT on ImageNet‑21k and fine‑tune.

---

## 5. Recommendations & Next Steps

### Recommended Model for Deployment
**Primary: CNNLSTM (8.75 M)**. It offers the best trade‑off between test accuracy (~52%, with potential for improvement) and parameter count. The unidirectional LSTM captures sufficient temporal context for 10‑frame gestures without over‑complicating.

**Alternative (if computation permits): TCN (3.09 M)** with added regularisation. It is lightweight and scalable; with proper tuning it could match or exceed the CNNLSTM.

### Concrete Improvements
1. **Regularisation**: Add Dropout (0.5) to LSTM inputs and FC layers. For TCN, add channel‑wise Dropout (Dropout2d) in residual blocks. Use weight decay (1e‑4) in AdamW.
2. **Data augmentation**: Random temporal jittering (frame dropping/duplication), random spatial cropping (64×64 from 80×80), horizontal flipping, slight colour jitter, and time masking (SpecAugment‑like).
3. **Learning rate schedule**: Cosine annealing with warm restarts, or reduce LR on plateau. Current constant lr=0.001 may be too high for some models.
4. **Label smoothing**: Reduces overconfidence; cross‑entropy with ε=0.1.
5. **Early stopping**: Based on validation loss, prevent overfitting after peak performance.
6. **For BiLSTM variants**: Reduce hidden size from 256 to 128 to cut parameters drastically; the small gain does not warrant 17M params.

### Hybrid Architecture Suggestion
- **CNN‑BiLSTM‑Attention (lightweight)**: Use a MobileNetV2‑style CNN backbone (pretrained) to extract per‑frame features, followed by a single‑layer BiLSTM with 128 units and temporal attention. Total parameters ~2M. This would leverage transfer learning, strong spatial features, and efficient temporal modelling.

### Data‑Efficiency and Scaling
- **Most data‑efficient**: CNNLSTM, because its CNN pretrained on ImageNet could be used (if allowed), and the LSTM learns from limited sequences.
- **Model that benefits most from more data**: ViT. With 10× more data (e.g., 20k clips) and aggressive augmentation, the ViT could learn meaningful attention patterns and surpass CNNs.
- **TCN** also scales well with more data because its inductive bias (temporal hierarchy) aligns with motion patterns.

---

## 6. Additional Observations

### Frame Flattening in TCN vs. Spatial Preservation
The TCN script flattens each 32×32×3 frame to a 3072‑D vector and applies a linear encoder. This completely discards spatial structure. While the linear layer can learn some spatial mixing, it is far less effective than a 2D CNN that preserves locality. This likely limits TCN’s ability to recognise fine‑grained hand shapes; a better design would use a small 2D CNN (e.g., 2–3 conv layers) per frame before the temporal convolutions. The current TCN is “spatially blind” – a significant weakness.

### Dataset Loading Pipeline
The `SequenceImageDataset` code allows directories to contain either direct images (single‑frame samples) or subdirectories of sequences. This mixed logic can accidentally treat single‑image folders as sequences of length 1, causing shape mismatches or misleading labels. **Recommendation**: Separate datasets explicitly, add assertion checks on frame count, and use a metadata file (CSV) with file paths and labels to avoid ambiguity.

### Edge‑Based Temporal Stream vs. Learned Flow
The Two‑Stream model uses a simple gradient magnitude edge map, which is crude and loses colour‑based cues. Optical flow (computed via TV‑L1 or a lightweight flownet) would provide dense motion information and dramatically improve the temporal stream. A better design: use a pretrained optical flow network to generate flow frames, then feed them into a 3D CNN or an LSTM. Alternatively, incorporate a learnable temporal stream that uses frame differences as input to a small ConvNet.

---

## Final Recommendation

Given the available data and task constraints, **deploy the CNNLSTM model with moderate regularisation (dropout, weight decay) and data augmentation**. Its parameter count (8.75 M) is manageable, its test accuracy of ~52% is competitive, and it converges stably. For resource‑constrained environments, explore the **TCN after adding a lightweight 2D CNN front‑end**; this could become the most efficient performer. Avoid the heavily overfitted CNN3D and the underperforming ViT unless significant additional data and pretraining are available. The research team should focus on improving data quality and diversity, and integrate temporal augmentations to boost all sequential models further.
