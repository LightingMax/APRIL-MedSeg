# 第 07 章：Foundation 模型

[English](07_foundation.md)

Foundation 模型是大规模预训练视觉 Transformer，提供丰富、可迁移的特征。UltimateMedSeg 集成 **35 个 Foundation 模型编码器**，覆盖 **9 个医学模态**，全部使用 **DPT（Dense Prediction Transformer）head** 进行多尺度特征提取。

---

## 架构：DPT Head

所有 Foundation ViT 使用 **DPT head** 而非朴素的 FPN：

```
预训练 ViT
  ├─ block_3  ──→ DPT stage 1 (1/4)
  ├─ block_7  ──→ DPT stage 2 (1/8)
  ├─ block_11 ──→ DPT stage 3 (1/16)
  └─ block_15 ──→ DPT stage 4 (1/32)
```

每个阶段从特定深度 block 提取特征，然后通过重组+融合层进行融合，产生比简单重塑 CLS token 更丰富的多尺度特征。

---

## 医学模态 — 9 大类别

### 1. 通用（5 个模型）

| 模型 | 预训练 | 用途 |
|------|--------|------|
| `dinov2_base/large/giant` | DINOv2 (ImageNet) | 通用迁移 |
| `dinov3_base` | DINOv3 | 最新自监督 |
| `dino_base` | DINO | 早期自监督 |
| `clip_vit_base` | CLIP (文本-图像) | 零样本能力 |
| `sam_vit_base/huge` | SAM (分割) | 分割迁移 |

```yaml
encoder:
  name: dinov2_base
  pretrained: true    # 自动下载权重
```

### 2. 病理（5 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `phikon` / `phikon_v2` | iBOT (组织学) | 组织病理学 |
| `uni` | DINOv2 (病理) | 通用病理 |
| `plip` | CLIP (病理) | 病理文本-图像 |
| `musk` | 多模态 (病理) | 多尺度病理 |

### 3. 放射（3 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `rad_dino` | DINO (放射) | 通用放射 |
| `omnirad` | 多模态 (放射) | 多任务放射 |
| `medsiglip` | SigLIP (医学) | 医学文本-图像 |

### 4. 眼科（4 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `retfound_dinov2` | MAE+DINOv2 (视网膜) | 视网膜疾病 |
| `retfound` | MAE (视网膜) | 视网膜成像 |
| `flair` | CLIP (视网膜) | 视网膜文本-图像 |
| `ophmae` | MAE (眼科) | 眼部成像 |

### 5. 皮肤（3 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `dermclip` | CLIP (皮肤镜) | 皮肤病变 |
| `monet` | 自监督 (皮肤) | 皮肤分割 |
| `panderm` | 多模态 (皮肤) | 综合皮肤病学 |

### 6. 多模态医学（3 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `biomedclip` | CLIP (生物医学) | 通用生物医学 |
| `medclip` | CLIP (医学) | 医学文本-图像 |
| `keep` | 多模态 (医学) | 知识增强 |

### 7. MLLM 视觉（8 个模型）

| 模型 | 基础 | 专长 |
|------|------|------|
| `qwen2_5_vl` | Qwen2.5-VL | 最新 VL 模型 |
| `qwen3_vl` | Qwen3-VL | 新一代 VL |
| `medgemma` | MedGemma | 医学 VL |
| `llava_med` | LLaVA-Med | 医学对话 |
| `huatuogpt` | HuatuoGPT | 中文医学 |
| `healthgpt` | HealthGPT | 健康领域 |
| `hulumed` | HuLuMed | 医学 VL |
| `lingshu` | LingShu | 中医 |

### 8. 超声（3 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `ultradino` | DINO (超声) | 超声特征 |
| `ultrafedfm` | 联邦学习 (超声) | 隐私保护 |
| `us_fmae` | MAE (超声) | 超声 MAE |

### 9. 内窥镜（1 个模型）

| 模型 | 预训练 | 专长 |
|------|--------|------|
| `endovit` | 自监督 (内窥镜) | 消化道内窥镜 |

---

## 微调策略

### 全量微调

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: false     # 训练所有编码器参数
```

### 冻结编码器 + 可训练解码器

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: true      # 冻结编码器，仅训练解码器
```

### 部分微调（最后 N 个 block）

```yaml
encoder:
  name: dinov2_base
  pretrained: true
  freeze: true
  params:
    unfreeze_last_n: 4    # 仅训练最后 4 个 block
```

---

## 权重管理

Foundation 模型权重**首次使用时自动下载**：

```bash
# 列出可下载的权重
python -m medseg.utils.weight_downloader list

# 下载指定权重
python -m medseg.utils.weight_downloader download dinov2_base

# 检查缓存
python -m medseg.utils.weight_downloader check
```

缓存路径：`~/.cache/ultimatemedseg/weights/`

---

## 总结

| 场景 | 推荐 Foundation 编码器 |
|------|----------------------|
| 通用医学 | `dinov2_base` |
| 病理 / 组织学 | `phikon_v2` 或 `uni` |
| 视网膜 / 眼科 | `retfound_dinov2` |
| 皮肤 | `panderm` |
| 放射 / CXR | `rad_dino` |
| 超声 | `ultradino` |
| 多模态 | `biomedclip` |
