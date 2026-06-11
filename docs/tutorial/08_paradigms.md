# Chapter 08: Advanced Training Paradigms

[中文文档](08_paradigms_CN.md)

Beyond standard supervised training, UltimateMedSeg supports **5 advanced training paradigms** with dedicated training scripts and YAML configurations.

---

## Overview

| Paradigm | Methods | Script | When to Use |
|----------|---------|--------|-------------|
| Semi-Supervised | 21 | `semi_train.py` | Limited labeled + abundant unlabeled data |
| Domain Adaptation | 18 | `train_domain_adaptation.py` | Source→target domain gap |
| Knowledge Distillation | 27 | `train_distillation.py` | Compress large model to small |
| Weakly Supervised | 28 | `train_weakly_supervised.py` | Coarse annotations (box, point, scribble) |
| Text-Guided | 13 | `train_text_guided.py` | Text prompts for segmentation |

---

## 1. Semi-Supervised Learning — 21 Methods

Use when you have a small labeled dataset and abundant unlabeled data.

### Consistency-Based

```yaml
# Mean Teacher: EMA teacher + consistency loss on unlabeled
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml
```

Methods: `mean_teacher`, `ua_mt`, `pi_model`, `temporal_ensembling`

### Pseudo-Labeling

```yaml
# CPS: Cross Pseudo Supervision between two networks
python semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml
```

Methods: `cps`, `fixmatch`, `flexmatch`, `freematch`, `softmatch`, `pseudo_label`, `corrmach`

### Co-Training

```yaml
# Cross-Teaching: two networks teach each other
python semi_train.py --config configs/training_paradigms/semi_supervision/cross_teaching.yaml
```

Methods: `cross_teaching`, `deep_co_training`, `ict`, `r_drop`

### Advanced

Methods: `unimatch`, `cct`, `urpc`, `allspark`, `diffrect`, `ssl4mis_u`

### Config Example

```yaml
model:
  num_classes: 4
  img_size: 224
  encoder: { name: timm_resnet50, pretrained: true }
  decoder: { name: unet }

semi:
  method: mean_teacher
  ema_decay: 0.99
  consistency_weight: 0.1
  labeled_ratio: 0.1        # 10% labeled data

data:
  type: synapse
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5

training:
  epochs: 200
  batch_size: 16
```

---

## 2. Domain Adaptation — 18 Methods

Use when training data (source) and deployment data (target) have different distributions.

### Entropy-Based

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --output_dir output/da_advent
```

Methods: `advent`, `dann`, `source_only`

### Test-Time Adaptation

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/tent.yaml \
    --output_dir output/da_tent
```

Methods: `tent`, `dpl`

### Style Transfer

Methods: `fda`, `pixmatch`, `crst`

### Advanced

Methods: `mic`, `daformer`, `hrda`, `pipa`, `ddb`, `sepicо`, `diga`, `micdrop`, `semivl`, `cbmt`

### Data Setup

```
data/
├── source/          # Source domain images + masks
│   ├── images/
│   └── masks/
├── target/          # Target domain images only (no masks)
│   └── images/
└── target_val/      # Target domain validation (with masks)
    ├── images/
    └── masks/
```

---

## 3. Knowledge Distillation — 27 Methods

Compress a large teacher model into a smaller student while preserving accuracy.

### Logit-Based

```bash
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5
```

Methods: `vanilla_kd`, `dkd`

### Feature-Based

Methods: `fitnets`, `at`, `fsp`, `nst`, `rkd`, `vid`

### Advanced

Methods: `mgd`, `dist`, `cirkd`, `cwd`, `reviewkd`, `simkd`, `norm`, `sdd`, `aicsd`, `lskd`, `ttm`, `ctkd`, `mlkd`

### Key Parameters

```yaml
distillation:
  type: logit          # logit / feature / attention
  temperature: 4.0     # softmax temperature
  alpha: 0.5           # balance: task loss vs distillation loss
```

---

## 4. Weakly Supervised — 28 Methods

Train with coarse annotations instead of pixel-level masks.

### Box Supervision

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box
```

Methods: `box`, `boxinst`

### Image-Level Labels (CAM-based)

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam
```

Methods: `cam`, `seam`, `puzzlecam`, `advcam`, `eps`, `recam`, `toco`, `lpcam`, `mars`, `bacon`, `wpgseg`, `dupl`, `more`, `psdpm`, `semple`

### Point / Scribble

Methods: `point`, `scribble`, `mil`, `em`

### Advanced

Methods: `gatedcrf`, `treeenergy`

---

## 5. Text-Guided — 13 Models + MLLM Pipeline

Use natural language text prompts to guide segmentation.

### Trainable Models

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/cris.yaml \
    --output_dir output/text_cris
```

Models: `cris`, `biomedparse`, `languidemedseg`, `lvit`, `tganet`, `tpro`, `causalclipseg`, `clip_universal`, `cxr_clip_seg`, `tp_drseg`, `medclip_sam`, `salip`, `medisee`

### MLLM Inference Pipeline

5 detectors × 4 segmenters = 20 combinations:

```python
from medseg.inference.mllm import MLLMPipeline

pipeline = MLLMPipeline(
    detector='grounding_dino',    # or qwen2_vl, qwen3_vl, internvl
    segmenter='sam2',             # or medsam, sam_med2d, lite_medsam
    text_prompt='liver tumor',
)
result = pipeline.predict(image_path)
```

---

## Running Paradigm Studies

```bash
# Semi-supervised comparison (6 methods)
bash scripts/experiments/run_semi_study.sh

# Domain adaptation comparison (8 methods)
bash scripts/experiments/run_da_study.sh

# Knowledge distillation comparison (7 methods)
bash scripts/experiments/run_kd_study.sh

# Weakly supervised comparison (6 methods)
bash scripts/experiments/run_weak_study.sh
```
