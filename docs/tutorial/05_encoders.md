# Chapter 05: Encoder Deep Dive

[中文文档](05_encoders_CN.md)

The encoder is the backbone of any segmentation model — it extracts multi-scale features from the input image. UltimateMedSeg provides **169 encoders** across 7 categories, plus a dynamic `timm_` wrapper that unlocks 1000+ additional models.

---

## Encoder Taxonomy

| Category | Count | Key Models | Best For |
|----------|-------|------------|----------|
| CNN | 12 | ResNet, ConvNeXt, EfficientNet, MedNeXt, R2U-Net | General-purpose, lightweight |
| Transformer | 18 | TransUNet, SwinUNet, PVTv2, MaxViT, MISSFormer | Long-range dependencies |
| Mamba / SSM | 10 | VMUNet, UMamba, LKM-UNet, LoG-VMamba | Linear-complexity sequence modeling |
| RWKV | 4 | RWKV-UNet, U-RWKV, MD-RWKV | Efficient RNN-like attention |
| Linear Attention | 5 | RetNet, Linformer, Performer, TTT, xLSTM | Sub-quadratic attention |
| KAN / MLP | 4 | UKAN, Rolling-UNet, UNeXt, Wav-KAN | Kolmogorov-Arnold networks |
| Foundation | 35 | DINOv2, CLIP-ViT, SAM-ViT, Phikon, RETFound | Transfer learning |

---

## 1. CNN Encoders

### Standard CNN Backbones

```yaml
# ResNet family
encoder:
  name: resnet50
  pretrained: true

# ConvNeXt (modern CNN)
encoder:
  name: convnext_base
  pretrained: true

# EfficientNet (compound scaling)
encoder:
  name: efficientnet_b4
  pretrained: true
```

### Medical-Specific CNNs

**MedNeXt** — 3D-capable CNN with large kernel convolutions:
```yaml
encoder:
  name: mednext_large
  pretrained: false
  params:
    kernel_size: 7
    deep_supervision: true
```

### Feature Map Dimensions

All CNN encoders return a list of multi-scale feature maps:
```
Input: (B, 3, 224, 224)
  → stage1: (B, 64, 112, 112)    # 1/2
  → stage2: (B, 128, 56, 56)     # 1/4
  → stage3: (B, 256, 28, 28)     # 1/8
  → stage4: (B, 512, 14, 14)     # 1/16
  → stage5: (B, 2048, 7, 7)      # 1/32
```

---

## 2. Transformer Encoders

### Vision Transformers

```yaml
# PVTv2 (Pyramid Vision Transformer)
encoder:
  name: pvt_v2_b2
  pretrained: true
  params:
    img_size: 224

# MaxViT (Multi-Axis Vision Transformer)
encoder:
  name: maxvit_small
  pretrained: true
```

### Hierarchical Transformers

**Swin Transformer** — Window-based attention with shifted windows:
```yaml
encoder:
  name: swin_tiny
  pretrained: true
  params:
    img_size: 224
    window_size: 7
```

### Key Design Choices

| Design | Models | Advantage |
|--------|--------|-----------|
| Global attention | ViT, TransUNet | Full context |
| Window attention | Swin, MaxViT | Linear complexity per window |
| Pyramid structure | PVTv2, SegFormer | Multi-scale features natively |
| Hybrid CNN+TF | TransUNet, MISSFormer | Local + global features |

---

## 3. Mamba / SSM Encoders

State Space Models provide linear-complexity sequence modeling, ideal for high-resolution medical images.

### VM-UNet (Visual Mamba)

```yaml
encoder:
  name: vmunet_tiny
  pretrained: false
  params:
    depths: [2, 2, 9, 2]
    dims: [96, 192, 384, 768]
```

### LKM-UNet (Large Kernel Mamba)

Combines large convolution kernels with Mamba for efficient feature extraction:
```yaml
encoder:
  name: lkm_unet
  pretrained: false
  params:
    large_kernel_size: 7
```

### When to Use Mamba

- **High-resolution inputs** (512×512+) — linear complexity scales well
- **Long-range dependencies** — global receptive field without quadratic cost
- **Memory-constrained deployment** — lower VRAM than Transformer

---

## 4. RWKV Encoders

RWKV combines the parallelizable training of Transformers with the efficient inference of RNNs.

```yaml
encoder:
  name: rwkv_unet
  pretrained: false
  params:
    depths: [2, 2, 2, 2]
    dims: [64, 128, 256, 512]
```

**Variants**: `rwkv_unet`, `u_rwkv`, `md_rwkv`, `rir_zigzag`

---

## 5. Dynamic timm Encoder

Any model from `timm.list_models()` can be used as an encoder by adding the `timm_` prefix:

```yaml
encoder:
  name: timm_efficientnet_b7
  pretrained: true

encoder:
  name: timm_convnextv2_tiny
  pretrained: true

encoder:
  name: timm_vit_large_patch16_224
  pretrained: true
```

### How It Works

1. The `timm_` prefix triggers the dynamic wrapper
2. The remaining name is passed to `timm.create_model()`
3. Feature extraction hooks are automatically attached at the correct stages
4. Output channels are auto-detected from a dummy forward pass

```python
import timm
print(len(timm.list_models()))  # 1000+ models available
```

---

## 6. Comparing Encoders

### Quick Comparison

```bash
# Run the SOTA benchmark across encoder types
bash scripts/experiments/run_sota_benchmark.sh
```

### Decoder Ablation with Different Encoders

```bash
# Compare 3 encoders × 15 decoders
bash scripts/experiments/run_decoder_study.sh
```

### Parameter Count Comparison

```python
from medseg.model_builder import build_model

encoders = ['resnet50', 'swin_tiny', 'pvt_v2_b2', 'vmunet_tiny']
for enc in encoders:
    cfg = {'model': {'encoder': {'name': enc}, 'decoder': {'name': 'unet'},
                      'num_classes': 9, 'img_size': 224}}
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{enc}: {params:.1f}M params")
```

---

## Summary

| Scenario | Recommended Encoder |
|----------|-------------------|
| Quick baseline | `timm_resnet50` |
| SOTA accuracy | `swin_tiny` or `pvt_v2_b2` |
| High-res input (512+) | `vmunet_tiny` or `lkm_unet` |
| Transfer learning | `dinov2_base` or `clip_vit_base` |
| Any architecture | `timm_*` (1000+ models) |
| Lightweight | `timm_efficientnet_b0` or `timm_mobilenetv3` |
