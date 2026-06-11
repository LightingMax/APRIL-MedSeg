# UltimateMedSeg 教程

[English](README.md)

以 **UltimateMedSeg** 框架为中心的深度学习医学图像分割实操教程系列。面向实验室内部使用，兼顾理论深度与工程实践。

---

## 学习路线

建议按顺序阅读，每章在前一章基础上递进。

```
01 概述 ──> 02 U-Net ──> 03 数据 ──> 04 训练
(是什么、为什么) (架构详解)  (管线配置) (优化实战)
```

---

## 教程索引

| 章节 | 标题 | 核心内容 |
|------|------|----------|
| [01](01_introduction_CN.md) | **医学图像分割概述** | 分割概念、临床意义、评价指标、方法演进、框架速览 |
| [02](02_unet_CN.md) | **U-Net 详解** | 编码器-解码器架构、跳跃连接、U-Net 家族、YAML 配置、训练命令 |
| [03](03_data_CN.md) | **数据与预处理** | 数据格式、目录约定、切分策略、增强管线、自定义数据集 |
| [04](04_training_CN.md) | **训练与评估** | 损失函数、优化器、学习率调度、AMP/DDP、评估流程、日志可视化 |
| [05](05_encoders_CN.md) | **编码器进阶** | CNN / Transformer / Mamba / RWKV 编码器对比、timm 动态编码器、特征提取 |
| [06](06_decoders_CN.md) | **解码器与跳跃连接** | CASCADE / EMCAD / Attention Gate、解码器消融实验、跳跃连接分类 |
| [07](07_foundation_CN.md) | **Foundation 模型** | 预训练 ViT 编码器、DPT head、微调策略、9 大医学模态 |
| [08](08_paradigms_CN.md) | **高级训练范式** | 半监督、域适应、知识蒸馏、弱监督、文本引导 |
| [09](09_deployment_CN.md) | **部署与推理** | ONNX 导出、TTA、集成推理、MLLM pipeline、模型性能分析 |

---

## 安装

```bash
git clone https://github.com/juntaoJianggavin/UltimateMedSeg.git
cd UltimateMedSeg

pip install -r requirements.txt
```

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `torch` | >= 2.0.0 | 深度学习框架 |
| `timm` | >= 0.9.0 | 编码器 backbone 库 |
| `monai` | >= 1.2.0 | 医学图像处理工具 |
| `albumentations` | >= 1.3.0 | 数据增强 |
| `einops` | >= 0.6.0 | 张量操作 |
| `tensorboard` | >= 2.13.0 | 训练可视化 |

---

## 快速开始

完成任意教程章节后，一行命令即可训练：

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml
```

通过命令行覆盖任意配置项：

```bash
python train.py --config configs/architectures/combinations/general/unet_basic.yaml \
    --override training.epochs=100 training.batch_size=8 model.num_classes=9
```

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [模型](../models/README_CN.md) | 169 编码器、40 解码器、128 完整网络 |
| [训练范式](../paradigms/README_CN.md) | 6 大训练范式 |
| [数据](../data/README.md) | 25 个数据集、增强管线 |
| [部署](../deployment/README.md) | ONNX 导出、TTA、集成推理 |
| [研究指南](../research_guide_CN.md) | 消融实验、基准测试 |
