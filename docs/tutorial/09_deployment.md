# Chapter 09: Deployment and Inference

[中文文档](09_deployment_CN.md)

This chapter covers the full inference pipeline: single-model evaluation, ensemble inference, test-time augmentation (TTA), ONNX export, and model profiling.

---

## 1. Single Model Inference

### Basic Evaluation

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth
```

Output includes per-class and mean Dice, IoU, and HD95.

### Save Predictions

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --save_pred --output_dir test_output/
```

Predictions are saved as `.npy` files in `test_output/predictions/`.

### Visualize Predictions

```bash
python scripts/visualize.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --input ./data/test/images/ \
    --output vis_output/
```

---

## 2. Ensemble Inference

Combine multiple checkpoints for better accuracy through logit averaging.

### Multi-Checkpoint Ensemble

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint ckpt_fold0.pth ckpt_fold1.pth ckpt_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit
```

### Averaging Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `logit` | Average raw logits before sigmoid/softmax | Multi-class segmentation |
| `softmax` | Average softmax probabilities | Stable multi-class |
| `sigmoid` | Average sigmoid probabilities | Binary / multi-label |

### How It Works

```
ckpt_0 → model_0 → logits_0 ─┐
ckpt_1 → model_1 → logits_1 ──┼→ weighted_avg → argmax → prediction
ckpt_2 → model_2 → logits_2 ─┘
```

---

## 3. Test-Time Augmentation (TTA)

Apply multiple augmentations at inference time and merge predictions for robustness.

### Basic TTA

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --tta \
    --tta-augs identity rot90 rot180 rot270 hflip vflip \
    --tta-merge mean
```

### Available Augmentations

| Name | Description |
|------|-------------|
| `identity` | No augmentation |
| `rot90`, `rot180`, `rot270` | Rotation |
| `hflip`, `vflip` | Horizontal / vertical flip |
| `brightness_up`, `brightness_down` | Brightness perturbation |
| `contrast_up`, `contrast_down` | Contrast perturbation |
| `gamma_up`, `gamma_down` | Gamma correction |

### Merge Strategies

| Strategy | Description |
|----------|-------------|
| `mean` | Average predictions |
| `gmean` | Geometric mean (better for probabilities) |
| `max` | Maximum confidence |
| `median` | Median (robust to outliers) |

### TTA + Ensemble Combined

TTA wraps the ensemble — ensemble runs first, then TTA:

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint ckpt_a.pth ckpt_b.pth \
    --ensemble-average logit \
    --tta --tta-merge mean
```

---

## 4. ONNX Export

Export models to ONNX format for deployment in non-Python environments.

### Basic Export

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx
```

### Export with Verification

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx --verify
```

The `--verify` flag runs both PyTorch and ONNX Runtime on the same input and compares outputs.

### Dynamic Input Size

```bash
python scripts/export_onnx.py \
    --config cfg.yaml --checkpoint best.pth \
    --output model.onnx \
    --dynamic
```

---

## 5. Model Profiling

Analyze computational cost with `profile_model.py`:

```bash
python profile_model.py --config configs/architectures/networks/general/transunet.yaml
```

Reports:
- **FLOPs** (floating-point operations)
- **Params** (total and trainable)
- **FPS** (inference speed, CUDA)

### Python API

```python
from fvcore.nn import FlopCountAnalysis
import torch

flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224))
print(f"FLOPs: {flops.total() / 1e9:.2f}G")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable / 1e6:.2f}M")
```

> Note: Frozen foundation encoder params are NOT counted as trainable.

---

## 6. MLLM Inference Pipeline

Use vision-language models for zero-shot segmentation:

```python
from medseg.inference.mllm import MLLMPipeline

# GroundingDINO + SAM2
pipeline = MLLMPipeline(
    detector='grounding_dino',
    segmenter='sam2',
    text_prompt='liver tumor',
)
result = pipeline.predict('data/test/image_001.png')
```

### Available Combinations (5 × 4 = 20)

| Detector | Models |
|----------|--------|
| `grounding_dino` | Grounding DINO |
| `qwen2_vl` | Qwen2-VL |
| `qwen2_5_vl` | Qwen2.5-VL |
| `qwen3_vl` | Qwen3-VL |
| `internvl` | InternVL |

| Segmenter | Models |
|-----------|--------|
| `sam2` | SAM 2 |
| `medsam` | MedSAM |
| `sam_med2d` | SAM-Med2D |
| `lite_medsam` | LiteMedSAM |

---

## Deployment Checklist

| Step | Command | Purpose |
|------|---------|---------|
| 1. Evaluate | `python test.py --config ... --checkpoint ...` | Baseline metrics |
| 2. Ensemble | `python test.py --checkpoint a b c --ensemble-average logit` | Boost accuracy |
| 3. TTA | `python test.py --tta --tta-merge mean` | Robustness |
| 4. Profile | `python profile_model.py --config ...` | Cost analysis |
| 5. Export | `python scripts/export_onnx.py --verify` | Deployment format |
| 6. Visualize | `python scripts/visualize.py --input ... --output ...` | Qualitative check |
