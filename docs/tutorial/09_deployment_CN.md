# 第 09 章：部署与推理

[English](09_deployment.md)

本章覆盖完整推理管线：单模型评估、集成推理、测试时增强（TTA）、ONNX 导出和模型性能分析。

---

## 1. 单模型推理

### 基础评估

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth
```

输出包含逐类和平均 Dice、IoU、HD95。

### 保存预测结果

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --save_pred --output_dir test_output/
```

预测结果以 `.npy` 格式保存在 `test_output/predictions/` 目录。

### 预测可视化

```bash
python scripts/visualize.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --input ./data/test/images/ \
    --output vis_output/
```

---

## 2. 集成推理

通过 logit 平均组合多个 checkpoint 以提升精度。

### 多 Checkpoint 集成

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint ckpt_fold0.pth ckpt_fold1.pth ckpt_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit
```

### 平均模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `logit` | 在 sigmoid/softmax 前平均原始 logits | 多类分割 |
| `softmax` | 平均 softmax 概率 | 稳定多类 |
| `sigmoid` | 平均 sigmoid 概率 | 二值/多标签 |

### 工作流程

```
ckpt_0 → model_0 → logits_0 ─┐
ckpt_1 → model_1 → logits_1 ──┼→ 加权平均 → argmax → 预测
ckpt_2 → model_2 → logits_2 ─┘
```

---

## 3. 测试时增强（TTA）

推理时对同一图像施加多种增强并合并预测，提升鲁棒性。

### 基础 TTA

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --tta \
    --tta-augs identity rot90 rot180 rot270 hflip vflip \
    --tta-merge mean
```

### 可用增强

| 名称 | 说明 |
|------|------|
| `identity` | 无增强 |
| `rot90`, `rot180`, `rot270` | 旋转 |
| `hflip`, `vflip` | 水平/垂直翻转 |
| `brightness_up`, `brightness_down` | 亮度扰动 |
| `contrast_up`, `contrast_down` | 对比度扰动 |
| `gamma_up`, `gamma_down` | Gamma 校正 |

### 合并策略

| 策略 | 说明 |
|------|------|
| `mean` | 平均预测 |
| `gmean` | 几何平均（更适合概率） |
| `max` | 最大置信度 |
| `median` | 中位数（对异常值鲁棒） |

### TTA + 集成 组合使用

TTA 包裹集成——集成先运行，然后 TTA：

```bash
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint ckpt_a.pth ckpt_b.pth \
    --ensemble-average logit \
    --tta --tta-merge mean
```

---

## 4. ONNX 导出

将模型导出为 ONNX 格式，用于非 Python 环境部署。

### 基础导出

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx
```

### 带验证的导出

```bash
python scripts/export_onnx.py \
    --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth \
    --output model.onnx --verify
```

`--verify` 标志会同时运行 PyTorch 和 ONNX Runtime 并比较输出。

### 动态输入尺寸

```bash
python scripts/export_onnx.py \
    --config cfg.yaml --checkpoint best.pth \
    --output model.onnx \
    --dynamic
```

---

## 5. 模型性能分析

使用 `profile_model.py` 分析计算开销：

```bash
python profile_model.py --config configs/architectures/networks/general/transunet.yaml
```

报告内容：
- **FLOPs**（浮点运算次数）
- **Params**（总参数量和可训练参数量）
- **FPS**（推理速度，CUDA）

### Python API

```python
from fvcore.nn import FlopCountAnalysis
import torch

flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224))
print(f"FLOPs: {flops.total() / 1e9:.2f}G")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练: {trainable / 1e6:.2f}M")
```

> 注意：冻结的 Foundation 编码器参数不计入可训练参数量。

---

## 6. MLLM 推理 Pipeline

使用视觉语言模型进行零样本分割：

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

### 可用组合（5 × 4 = 20 种）

| 检测器 | 模型 |
|--------|------|
| `grounding_dino` | Grounding DINO |
| `qwen2_vl` | Qwen2-VL |
| `qwen2_5_vl` | Qwen2.5-VL |
| `qwen3_vl` | Qwen3-VL |
| `internvl` | InternVL |

| 分割器 | 模型 |
|--------|------|
| `sam2` | SAM 2 |
| `medsam` | MedSAM |
| `sam_med2d` | SAM-Med2D |
| `lite_medsam` | LiteMedSAM |

---

## 部署检查清单

| 步骤 | 命令 | 目的 |
|------|------|------|
| 1. 评估 | `python test.py --config ... --checkpoint ...` | 基线指标 |
| 2. 集成 | `python test.py --checkpoint a b c --ensemble-average logit` | 提升精度 |
| 3. TTA | `python test.py --tta --tta-merge mean` | 鲁棒性 |
| 4. 分析 | `python profile_model.py --config ...` | 成本分析 |
| 5. 导出 | `python scripts/export_onnx.py --verify` | 部署格式 |
| 6. 可视化 | `python scripts/visualize.py --input ... --output ...` | 定性检查 |
