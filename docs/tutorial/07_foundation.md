# Chapter 07: Foundation Models

[中文文档](07_foundation_CN.md)

Foundation models are large-scale pre-trained vision transformers that provide rich, transferable features. UltimateMedSeg integrates **35 foundation model encoders** across **9 medical modalities**, all using the **DPT (Dense Prediction Transformer) head** for multi-scale feature extraction.

---

## Architecture: DPT Head

All foundation ViTs use a **DPT head** instead of naive FPN:

```
Pre-trained ViT
  ├─ block_3  ──→ DPT stage 1 (1/4)
  ├─ block_7  ──→ DPT stage 2 (1/8)
  ├─ block_11 ──→ DPT stage 3 (1/16)
  └─ block_15 ──→ DPT stage 4 (1/32)
```

Each stage extracts features from a specific depth block, then fuses them through reassemble + fusion layers. This produces richer multi-scale features than simply reshaping CLS tokens.

---

## Medical Modalities — 9 Categories

### 1. General (5 models)

Pre-trained on large-scale natural image datasets:

| Model | Pre-training | Use Case |
|-------|-------------|----------|
| `dinov2_base/large/giant` | DINOv2 (ImageNet) | General transfer |
| `dinov3_base` | DINOv3 | Latest self-supervised |
| `dino_base` | DINO | Earlier self-supervised |
| `clip_vit_base` | CLIP (text-image) | Zero-shot capability |
| `sam_vit_base/huge` | SAM (segmentation) | Segmentation transfer |

```yaml
encoder:
  name: dinov2_base
  pretrained: true    # auto-downloads weights
```

### 2. Pathology (5 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `phikon` / `phikon_v2` | iBOT (histology) | Histopathology |
| `uni` | DINOv2 (pathology) | Universal pathology |
| `plip` | CLIP (pathology) | Pathology text-image |
| `musk` | Multi-modal (pathology) | Multi-scale pathology |

```yaml
encoder:
  name: phikon_v2
  pretrained: true
```

### 3. Radiology (3 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `rad_dino` | DINO (radiology) | General radiology |
| `omnirad` | Multi-modal (radiology) | Multi-task radiology |
| `medsiglip` | SigLIP (medical) | Medical text-image |

### 4. Ophthalmology (4 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `retfound_dinov2` | MAE+DINOv2 (retinal) | Retinal disease |
| `retfound` | MAE (retinal) | Retinal imaging |
| `flair` | CLIP (retinal) | Retinal text-image |
| `ophmae` | MAE (ophthalmology) | Eye imaging |

### 5. Dermatology (3 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `dermclip` | CLIP (dermoscopy) | Skin lesion |
| `monet` | Self-supervised (skin) | Skin segmentation |
| `panderm` | Multi-modal (skin) | Comprehensive dermatology |

### 6. Multimodal Medical (3 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `biomedclip` | CLIP (biomedical) | General biomedical |
| `medclip` | CLIP (medical) | Medical text-image |
| `keep` | Multi-modal (medical) | Knowledge-enhanced |

### 7. MLLM Vision (8 models)

Large vision-language models that provide rich visual understanding:

| Model | Base | Specialty |
|-------|------|-----------|
| `qwen2_5_vl` | Qwen2.5-VL | Latest VL model |
| `qwen3_vl` | Qwen3-VL | Next-gen VL |
| `medgemma` | MedGemma | Medical VL |
| `llava_med` | LLaVA-Med | Medical conversation |
| `huatuogpt` | HuatuoGPT | Chinese medical |
| `healthgpt` | HealthGPT | Health domain |
| `hulumed` | HuLuMed | Medical VL |
| `lingshu` | LingShu | Traditional Chinese medicine |

### 8. Ultrasound (3 models)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `ultradino` | DINO (ultrasound) | Ultrasound features |
| `ultrafedfm` | Federated (ultrasound) | Privacy-preserving |
| `us_fmae` | MAE (ultrasound) | Ultrasound MAE |

### 9. Endoscopy (1 model)

| Model | Pre-training | Specialty |
|-------|-------------|-----------|
| `endovit` | Self-supervised (endoscopy) | GI endoscopy |

---

## Fine-Tuning Strategies

### Full Fine-Tuning

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: false     # train all encoder params
```

### Frozen Encoder + Trainable Decoder

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: true      # freeze encoder, train decoder only
```

### Partial Fine-Tuning (last N blocks)

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: true
  params:
    unfreeze_last_n: 4    # only train last 4 blocks
```

---

## Weight Management

Foundation model weights are **automatically downloaded** on first use:

```bash
# List downloadable weights
python -m medseg.utils.weight_downloader list

# Download specific weights
python -m medseg.utils.weight_downloader download dinov2_base

# Check cache
python -m medseg.utils.weight_downloader check
```

Cached at `~/.cache/ultimatemedseg/weights/`.

---

## Comparing Foundation vs. Standard Encoders

```bash
# Foundation model benchmark (9 modalities × 35 encoders)
bash scripts/experiments/run_foundation_benchmark.sh
```

```yaml
# Compare: DINOv2 vs ResNet50
# DINOv2
model:
  encoder: { name: dinov2_base, pretrained: true }
  decoder: { name: unet }

# ResNet50
model:
  encoder: { name: timm_resnet50, pretrained: true }
  decoder: { name: unet }
```

---

## Summary

| Scenario | Recommended Foundation Encoder |
|----------|-------------------------------|
| General medical | `dinov2_base` |
| Pathology / histology | `phikon_v2` or `uni` |
| Retinal / ophthalmology | `retfound_dinov2` |
| Dermatology | `panderm` |
| Radiology / CXR | `rad_dino` |
| Ultrasound | `ultradino` |
| Multi-modal | `biomedclip` |
