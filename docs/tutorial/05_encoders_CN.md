# 第 05 章：编码器进阶

[English](05_encoders.md)

编码器是分割模型的核心骨干——负责从输入图像中提取多尺度特征。UltimateMedSeg 提供 **169 个编码器**，涵盖 7 大类别，加上动态 `timm_` 封装器可解锁 1000+ 额外模型。

---

## 编码器分类

| 类别 | 数量 | 代表模型 | 适用场景 |
|------|------|----------|----------|
| CNN | 12 | ResNet, ConvNeXt, EfficientNet, MedNeXt, R2U-Net | 通用、轻量化 |
| Transformer | 18 | TransUNet, SwinUNet, PVTv2, MaxViT, MISSFormer | 长距离依赖 |
| Mamba / SSM | 10 | VMUNet, UMamba, LKM-UNet, LoG-VMamba | 线性复杂度序列建模 |
| RWKV | 4 | RWKV-UNet, U-RWKV, MD-RWKV | 高效类 RNN 注意力 |
| 线性注意力 | 5 | RetNet, Linformer, Performer, TTT, xLSTM | 亚二次复杂度注意力 |
| KAN / MLP | 4 | UKAN, Rolling-UNet, UNeXt, Wav-KAN | Kolmogorov-Arnold 网络 |
| Foundation | 35 | DINOv2, CLIP-ViT, SAM-ViT, Phikon, RETFound | 迁移学习 |

---

## 1. CNN 编码器

### 标准 CNN 骨干

```yaml
# ResNet 系列
encoder:
  name: resnet50
  pretrained: true

# ConvNeXt（现代 CNN）
encoder:
  name: convnext_base
  pretrained: true

# EfficientNet（复合缩放）
encoder:
  name: efficientnet_b4
  pretrained: true
```

### 医学专用 CNN

**MedNeXt** — 大核卷积的 3D 兼容 CNN：
```yaml
encoder:
  name: mednext_large
  pretrained: false
  params:
    kernel_size: 7
    deep_supervision: true
```

### 特征图维度

所有 CNN 编码器返回多尺度特征图列表：
```
输入: (B, 3, 224, 224)
  → stage1: (B, 64, 112, 112)    # 1/2
  → stage2: (B, 128, 56, 56)     # 1/4
  → stage3: (B, 256, 28, 28)     # 1/8
  → stage4: (B, 512, 14, 14)     # 1/16
  → stage5: (B, 2048, 7, 7)      # 1/32
```

---

## 2. Transformer 编码器

### 视觉 Transformer

```yaml
# PVTv2（金字塔视觉 Transformer）
encoder:
  name: pvt_v2_b2
  pretrained: true
  params:
    img_size: 224

# MaxViT（多轴视觉 Transformer）
encoder:
  name: maxvit_small
  pretrained: true
```

### 层次化 Transformer

**Swin Transformer** — 基于窗口的注意力与移动窗口：
```yaml
encoder:
  name: swin_tiny
  pretrained: true
  params:
    img_size: 224
    window_size: 7
```

### 关键设计选择

| 设计 | 模型 | 优势 |
|------|------|------|
| 全局注意力 | ViT, TransUNet | 完整上下文 |
| 窗口注意力 | Swin, MaxViT | 窗口内线性复杂度 |
| 金字塔结构 | PVTv2, SegFormer | 原生多尺度特征 |
| 混合 CNN+TF | TransUNet, MISSFormer | 局部+全局特征 |

---

## 3. Mamba / SSM 编码器

状态空间模型提供线性复杂度序列建模，非常适合高分辨率医学图像。

### VM-UNet（视觉 Mamba）

```yaml
encoder:
  name: vmunet_tiny
  pretrained: false
  params:
    depths: [2, 2, 9, 2]
    dims: [96, 192, 384, 768]
```

### LKM-UNet（大核 Mamba）

结合大卷积核与 Mamba 实现高效特征提取：
```yaml
encoder:
  name: lkm_unet
  pretrained: false
  params:
    large_kernel_size: 7
```

### 何时使用 Mamba

- **高分辨率输入**（512×512+）— 线性复杂度优势明显
- **长距离依赖** — 无需二次复杂度即可获得全局感受野
- **显存受限部署** — 比 Transformer 更省 VRAM

---

## 4. RWKV 编码器

RWKV 结合了 Transformer 的可并行训练优势与 RNN 的高效推理能力。

```yaml
encoder:
  name: rwkv_unet
  pretrained: false
  params:
    depths: [2, 2, 2, 2]
    dims: [64, 128, 256, 512]
```

**变体**: `rwkv_unet`, `u_rwkv`, `md_rwkv`, `rir_zigzag`

---

## 5. 动态 timm 编码器

任何 `timm.list_models()` 中的模型加 `timm_` 前缀即可作为编码器使用：

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

### 工作原理

1. `timm_` 前缀触发动态封装器
2. 剩余名称传递给 `timm.create_model()`
3. 特征提取钩子自动挂载到正确阶段
4. 输出通道通过 dummy forward 自动检测

```python
import timm
print(len(timm.list_models()))  # 1000+ 模型可用
```

---

## 6. 编码器对比

### 快速对比

```bash
# 跨编码器类型运行 SOTA 基准测试
bash scripts/experiments/run_sota_benchmark.sh
```

### 解码器消融 + 不同编码器

```bash
# 对比 3 编码器 × 15 解码器
bash scripts/experiments/run_decoder_study.sh
```

### 参数量对比

```python
from medseg.model_builder import build_model

encoders = ['resnet50', 'swin_tiny', 'pvt_v2_b2', 'vmunet_tiny']
for enc in encoders:
    cfg = {'model': {'encoder': {'name': enc}, 'decoder': {'name': 'unet'},
                      'num_classes': 9, 'img_size': 224}}
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{enc}: {params:.1f}M 参数")
```

---

## 总结

| 场景 | 推荐编码器 |
|------|-----------|
| 快速基线 | `timm_resnet50` |
| SOTA 精度 | `swin_tiny` 或 `pvt_v2_b2` |
| 高分辨率输入 (512+) | `vmunet_tiny` 或 `lkm_unet` |
| 迁移学习 | `dinov2_base` 或 `clip_vit_base` |
| 任意架构 | `timm_*`（1000+ 模型） |
| 轻量化 | `timm_efficientnet_b0` 或 `timm_mobilenetv3` |
