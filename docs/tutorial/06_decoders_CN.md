# 第 06 章：解码器与跳跃连接

[English](06_decoders.md)

解码器从编码器的压缩特征图中重建像素级预测。跳跃连接通过融合细粒度空间细节与高层语义来弥合编码器-解码器之间的鸿沟。UltimateMedSeg 提供 **40 个解码器** 和 **25 个跳跃连接**。

---

## 解码器分类

| 类别 | 数量 | 代表模型 | 核心思想 |
|------|------|----------|----------|
| 基础 | 4 | UNet, Bilinear, Deconv, DepthwiseSep | 标准上采样 |
| 密集连接 | 2 | UNet++, UNet3+ | 密集多尺度连接 |
| 级联 | 10 | CASCADE, EMCAD, G-CASCADE, MERIT | 迭代精炼 |
| 注意力 | 3 | AG, HAM, Lawin | 注意力引导上采样 |
| Transformer | 5 | DAEFormer, MTUNet, nnFormer | 基于 Transformer 解码 |
| MLP | 2 | SegFormer MLP, MLP 解码器 | 轻量化 MLP 上采样 |
| 网络专属 | 12 | TransUNet CUP, HiFormer, H2Former | 网络专属设计 |
| 金字塔 | 1 | UPerNet | 特征金字塔池化 |
| Mamba | 1 | VM-UNet | 基于 SSM 解码 |

---

## 1. 基础解码器

### UNet 解码器

经典解码器，使用转置卷积上采样和跳跃拼接：

```yaml
decoder:
  name: unet
  params: {}
```

### Bilinear 解码器

简单的双线性插值——快速且有效：

```yaml
decoder:
  name: bilinear
  params: {}
```

---

## 2. 密集连接解码器

### UNet++（嵌套 UNet）

跨所有分辨率级别的密集跳跃连接：

```yaml
decoder:
  name: unet_pp
  params:
    deep_supervision: true
```

### UNet3+

全尺度跳跃连接——每个解码器级别接收来自所有编码器级别的特征：

```yaml
decoder:
  name: unet3plus
  params: {}
```

---

## 3. 级联解码器

级联解码器通过多个解码阶段迭代精炼预测结果。

### CASCADE

多阶段级联，每个阶段进行特征精炼：

```yaml
decoder:
  name: cascade
  params:
    num_stages: 4
```

### EMCAD

高效多尺度级联注意力解码器：

```yaml
decoder:
  name: emcad
  params: {}
```

---

## 4. 注意力解码器

### Attention Gate (AG)

对跳跃特征施加软注意力门控——聚焦相关空间区域：

```yaml
decoder:
  name: attention_gate
  params: {}
skip_connection:
  name: ag
```

### HAM（混合注意力模块）

在解码器中结合空间注意力和通道注意力：

```yaml
decoder:
  name: ham
  params: {}
```

---

## 5. 解码器消融实验

UltimateMedSeg 提供系统化的消融实验框架：

```bash
# 3 编码器 × 15 经典解码器
bash scripts/experiments/run_decoder_study.sh
```

---

## 跳跃连接 — 25 种方法

### 分类

| 类别 | 数量 | 方法 |
|------|------|------|
| 基础 | 2 | `concat`, `dense` |
| 注意力 | 10 | `ag`, `cab`, `sab`, `scse`, `cbam`, `gating`, `gru`, `gab`, `sc_att`, `ta_mosc` |
| Transformer | 5 | `cross_attn`, `trans_fusion`, `agg_attn`, `miss_former`, `uctrans` |
| Mamba | 1 | `sk_vm_pp` |
| 融合 | 6 | `bi_fusion`, `deformable`, `multi_scale`, `feature_refine`, `ccm`, `sdi` |

### 基础跳跃：拼接

```yaml
skip_connection:
  name: concat
```

### 注意力跳跃：CBAM

对跳跃特征施加通道和空间注意力：

```yaml
skip_connection:
  name: cbam
  params:
    reduction: 16
```

### 跳跃连接消融

```bash
# 3 编码器 × 12 跳跃连接
bash scripts/experiments/run_skip_study.sh
```

---

## 自由组合解码器与跳跃

模块化系统允许自由组合：

```yaml
model:
  num_classes: 9
  img_size: 224
  encoder:
    name: timm_resnet50
    pretrained: true
  decoder:
    name: cascade
    params:
      num_stages: 4
  skip_connection:
    name: cbam
    params:
      reduction: 16
  bottleneck:
    name: aspp
```

### 关键兼容性规则

1. **级联解码器** — `skip_features` 排除瓶颈层通道（仅编码器中间特征）
2. **网络专属解码器**（如 `transunet`, `hiformer`）— 需匹配编码器；忽略 `skip_connection`
3. **has_internal_skip** — 部分解码器（UNet++, UCTransNet）自行管理跳跃连接

---

## 总结

| 场景 | 推荐解码器 | 跳跃连接 |
|------|-----------|----------|
| 快速基线 | `unet` | `concat` |
| SOTA 精度 | `cascade` 或 `emcad` | `cbam` 或 `cross_attn` |
| 轻量化 | `bilinear` | `concat` |
| 密集特征 | `unet_pp` 或 `unet3plus` | （内部） |
| 注意力引导 | `attention_gate` | `ag` |
