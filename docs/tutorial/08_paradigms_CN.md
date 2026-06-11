# 第 08 章：高级训练范式

[English](08_paradigms.md)

除标准监督训练外，UltimateMedSeg 支持 **5 大高级训练范式**，各有专用训练脚本和 YAML 配置。

---

## 概述

| 范式 | 方法数 | 脚本 | 适用场景 |
|------|--------|------|----------|
| 半监督 | 21 | `semi_train.py` | 少量标注 + 大量无标注数据 |
| 域适应 | 18 | `train_domain_adaptation.py` | 源域→目标域分布差异 |
| 知识蒸馏 | 27 | `train_distillation.py` | 大模型压缩到小模型 |
| 弱监督 | 28 | `train_weakly_supervised.py` | 粗糙标注（框、点、涂鸦） |
| 文本引导 | 13 | `train_text_guided.py` | 文本提示引导分割 |

---

## 1. 半监督学习 — 21 个方法

适用于少量标注数据 + 大量无标注数据的场景。

### 一致性正则化

```bash
# Mean Teacher: EMA 教师 + 无标注数据一致性损失
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml
```

方法: `mean_teacher`, `ua_mt`, `pi_model`, `temporal_ensembling`

### 伪标签

```bash
# CPS: 双网络交叉伪监督
python semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml
```

方法: `cps`, `fixmatch`, `flexmatch`, `freematch`, `softmatch`, `pseudo_label`, `corrmach`

### 协同训练

方法: `cross_teaching`, `deep_co_training`, `ict`, `r_drop`

### 高级方法

方法: `unimatch`, `cct`, `urpc`, `allspark`, `diffrect`, `ssl4mis_u`

### 配置示例

```yaml
semi:
  method: mean_teacher
  ema_decay: 0.99
  consistency_weight: 0.1
  labeled_ratio: 0.1        # 10% 标注数据

data:
  type: synapse
  train_dir: ./data/Synapse/train_npz
  val_dir: ./data/Synapse/test_vol_h5
```

---

## 2. 域适应 — 18 个方法

适用于训练数据（源域）和部署数据（目标域）分布不同的场景。

### 基于熵

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --output_dir output/da_advent
```

方法: `advent`, `dann`, `source_only`

### 测试时自适应

```bash
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/tent.yaml \
    --output_dir output/da_tent
```

方法: `tent`, `dpl`

### 风格迁移

方法: `fda`, `pixmatch`, `crst`

### 高级方法

方法: `mic`, `daformer`, `hrda`, `pipa`, `ddb`, `sepicо`, `diga`, `micdrop`, `semivl`, `cbmt`

### 数据目录结构

```
data/
├── source/          # 源域：图像 + 标注
│   ├── images/
│   └── masks/
├── target/          # 目标域：仅图像（无标注）
│   └── images/
└── target_val/      # 目标域验证集（含标注）
    ├── images/
    └── masks/
```

---

## 3. 知识蒸馏 — 27 个方法

将大型教师模型压缩为小型学生模型，同时保持精度。

### 基于 Logit

```bash
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5
```

方法: `vanilla_kd`, `dkd`

### 基于特征

方法: `fitnets`, `at`, `fsp`, `nst`, `rkd`, `vid`

### 高级方法

方法: `mgd`, `dist`, `cirkd`, `cwd`, `reviewkd`, `simkd`, `norm`, `sdd`, `aicsd`, `lskd`, `ttm`, `ctkd`, `mlkd`

### 关键参数

```yaml
distillation:
  type: logit          # logit / feature / attention
  temperature: 4.0     # softmax 温度
  alpha: 0.5           # 平衡：任务损失 vs 蒸馏损失
```

---

## 4. 弱监督 — 28 个方法

使用粗糙标注代替像素级 mask 进行训练。

### 边界框监督

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
    --supervision_type box
```

方法: `box`, `boxinst`

### 图像级标签（基于 CAM）

```bash
python train_weakly_supervised.py \
    --config configs/training_paradigms/weak_supervision/cam.yaml \
    --supervision_type cam
```

方法: `cam`, `seam`, `puzzlecam`, `advcam`, `eps`, `recam`, `toco`, `lpcam`, `mars`, `bacon`, `wpgseg`, `dupl`, `more`, `psdpm`, `semple`

### 点/涂鸦标注

方法: `point`, `scribble`, `mil`, `em`

---

## 5. 文本引导 — 13 个模型 + MLLM Pipeline

使用自然语言文本提示引导分割。

### 可训练模型

```bash
python train_text_guided.py \
    --config configs/training_paradigms/text_guided/cris.yaml \
    --output_dir output/text_cris
```

模型: `cris`, `biomedparse`, `languidemedseg`, `lvit`, `tganet`, `tpro`, `causalclipseg`, `clip_universal`, `cxr_clip_seg`, `tp_drseg`, `medclip_sam`, `salip`, `medisee`

### MLLM 推理 Pipeline

5 检测器 × 4 分割器 = 20 种组合：

```python
from medseg.inference.mllm import MLLMPipeline

pipeline = MLLMPipeline(
    detector='grounding_dino',    # 或 qwen2_vl, qwen3_vl, internvl
    segmenter='sam2',             # 或 medsam, sam_med2d, lite_medsam
    text_prompt='liver tumor',
)
result = pipeline.predict(image_path)
```

---

## 范式对比实验

```bash
# 半监督对比（6 方法）
bash scripts/experiments/run_semi_study.sh

# 域适应对比（8 方法）
bash scripts/experiments/run_da_study.sh

# 知识蒸馏对比（7 方法）
bash scripts/experiments/run_kd_study.sh

# 弱监督对比（6 方法）
bash scripts/experiments/run_weak_study.sh
```
