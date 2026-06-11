# Chapter 06: Decoders and Skip Connections

[中文文档](06_decoders_CN.md)

The decoder reconstructs pixel-level predictions from the encoder's compressed feature maps. Skip connections bridge the encoder-decoder gap by fusing fine-grained spatial details with high-level semantics. UltimateMedSeg provides **40 decoders** and **25 skip connections**.

---

## Decoder Taxonomy

| Category | Count | Examples | Key Idea |
|----------|-------|----------|----------|
| Basic | 4 | UNet, Bilinear, Deconv, DepthwiseSep | Standard upsampling |
| Dense | 2 | UNet++, UNet3+ | Dense multi-scale connections |
| Cascade | 10 | CASCADE, EMCAD, G-CASCADE, MERIT | Iterative refinement |
| Attention | 3 | AG, HAM, Lawin | Attention-guided upsampling |
| Transformer | 5 | DAEFormer, MTUNet, nnFormer | Transformer-based decoding |
| MLP | 2 | SegFormer MLP, MLP Decoder | Lightweight MLP upsampling |
| Specific | 12 | TransUNet CUP, HiFormer, H2Former | Network-specific designs |
| Pyramid | 1 | UPerNet | Feature pyramid pooling |
| Mamba | 1 | VM-UNet | SSM-based decoding |

---

## 1. Basic Decoders

### UNet Decoder

The classic decoder with transposed convolution upsampling and skip concatenation:

```yaml
decoder:
  name: unet
  params: {}
```

### Bilinear Decoder

Simple bilinear interpolation — fast and effective:

```yaml
decoder:
  name: bilinear
  params: {}
```

### Deconv Decoder

Learnable upsampling via transposed convolutions:

```yaml
decoder:
  name: deconv
  params: {}
```

---

## 2. Dense Connection Decoders

### UNet++ (Nested UNet)

Dense skip connections across all resolution levels:

```yaml
decoder:
  name: unet_pp
  params:
    deep_supervision: true
```

### UNet3+

Full-scale skip connections — each decoder level receives features from ALL encoder levels:

```yaml
decoder:
  name: unet3plus
  params: {}
```

---

## 3. Cascade Decoders

Cascade decoders iteratively refine predictions through multiple decoding stages.

### CASCADE

Multi-stage cascade with feature refinement at each stage:

```yaml
decoder:
  name: cascade
  params:
    num_stages: 4
```

### EMCAD

Efficient multi-scale cascade with attention:

```yaml
decoder:
  name: emcad
  params: {}
```

### G-CASCADE

Gated cascade — uses gating mechanisms between stages:

```yaml
decoder:
  name: g_cascade
  params:
    num_stages: 3
```

### MERIT

Multi-scale encoder-refined iterative transformer decoder:

```yaml
decoder:
  name: merit
  params: {}
```

---

## 4. Attention Decoders

### Attention Gate (AG)

Soft attention gates on skip features — focuses on relevant spatial regions:

```yaml
decoder:
  name: attention_gate
  params: {}
skip_connection:
  name: ag
```

### HAM (Hybrid Attention Module)

Combines spatial and channel attention in the decoder:

```yaml
decoder:
  name: ham
  params: {}
```

### Lawin

Lightweight attention-based decoder with large kernel convolutions:

```yaml
decoder:
  name: lawin
  params: {}
```

---

## 5. Transformer Decoders

### DAEFormer

Dual attention engine — spatial + channel attention in transformer decoder:

```yaml
decoder:
  name: daeformer
  params:
    embed_dim: 256
```

### UCTransNet

Cross-attention between encoder and decoder features:

```yaml
decoder:
  name: uctransnet
  params: {}
```

---

## 6. Decoder Ablation Study

UltimateMedSeg provides a systematic ablation framework:

```bash
# 3 encoders × 15 classic decoders
bash scripts/experiments/run_decoder_study.sh
```

### YAML Configuration for Ablation

```yaml
# ablation_resnet50_cascade.yaml
_base_: ../base_resnet50.yaml
model:
  decoder:
    name: cascade
    params:
      num_stages: 4

# ablation_resnet50_unetpp.yaml
_base_: ../base_resnet50.yaml
model:
  decoder:
    name: unet_pp
```

---

## Skip Connections — 25 Methods

Skip connections fuse encoder features with decoder upsampled features at corresponding resolution levels.

### Taxonomy

| Category | Count | Methods |
|----------|-------|---------|
| Basic | 2 | `concat`, `dense` |
| Attention | 10 | `ag`, `cab`, `sab`, `scse`, `cbam`, `gating`, `gru`, `gab`, `sc_att`, `ta_mosc` |
| Transformer | 5 | `cross_attn`, `trans_fusion`, `agg_attn`, `miss_former`, `uctrans` |
| Mamba | 1 | `sk_vm_pp` |
| Fusion | 6 | `bi_fusion`, `deformable`, `multi_scale`, `feature_refine`, `ccm`, `sdi` |

### Basic Skip: Concatenation

```yaml
skip_connection:
  name: concat
```

### Attention Skip: CBAM

Channel and spatial attention on skip features:

```yaml
skip_connection:
  name: cbam
  params:
    reduction: 16
```

### Transformer Skip: Cross-Attention

```yaml
skip_connection:
  name: cross_attn
  params:
    num_heads: 8
```

### Skip Connection Ablation

```bash
# 3 encoders × 12 skip connections
bash scripts/experiments/run_skip_study.sh
```

---

## Combining Decoders and Skips

The modular system allows free combination:

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

### Key Compatibility Rules

1. **Cascade decoders** — `skip_features` exclude the bottleneck channel (only encoder intermediate features)
2. **Network-specific decoders** (e.g. `transunet`, `hiformer`) — require matching encoder; ignore `skip_connection`
3. **has_internal_skip** — some decoders (UNet++, UCTransNet) manage their own skip connections

---

## Summary

| Scenario | Recommended Decoder | Skip Connection |
|----------|-------------------|-----------------|
| Quick baseline | `unet` | `concat` |
| SOTA accuracy | `cascade` or `emcad` | `cbam` or `cross_attn` |
| Lightweight | `bilinear` | `concat` |
| Dense features | `unet_pp` or `unet3plus` | (internal) |
| Attention-guided | `attention_gate` | `ag` |
