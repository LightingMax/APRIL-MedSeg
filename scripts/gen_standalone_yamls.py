"""Generate architecture-key YAML configs for all standalone networks.

Creates 3 configs per model:
  - configs/acdc/{model}.yaml       (img_size=224, num_classes=4, 200 epochs, lr=1e-4)
  - configs/synapse/{model}.yaml    (img_size=224, num_classes=9, 200 epochs, lr=1e-4)
  - configs/binary/{model}.yaml     (img_size=256, num_classes=2, 300 epochs, lr=1e-3)

Skips models that already have a YAML in the target directory.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "segmentation_tool", "configs")

# Models to generate YAMLs for (arch_key -> description comment)
MODELS = {
    # CNN
    "unet3plus":         "UNet3+: dense skip connections across all encoder-decoder levels",
    "lv_unet":           "LV-UNet: lightweight volumetric UNet",
    "ege_unet":          "EGE-UNet: enhanced global encoder UNet",
    "malunet":           "MALUNet: multi-attention lightweight UNet",
    "lite_unet":         "Lite-UNet: lightweight efficient UNet",
    "mk_unet":           "MK-UNet: multi-kernel UNet",
    "u_lite":            "U-Lite: ultra-lightweight segmentation",
    "ultralbm_unet":     "UltraLBM-UNet: ultra-lightweight block module UNet",
    "acc_unet":          "ACC-UNet: attention-connected context UNet",
    "cmunext":           "CMUNeXt: large kernel CNN + skip fusion",
    "multiresunet":      "MultiResUNet: multi-resolution UNet with ResPath",
    "scseunet":          "SCSE-UNet: spatial+channel squeeze-excitation UNet",
    "resunet_a":         "ResUNet-a: deep residual UNet for medical segmentation",
    "sa_unet":           "SA-UNet: spatial attention UNet",
    "pan":               "PAN: pyramid attention network for segmentation",
    "denseunet":         "DenseUNet: DenseNet-based UNet",
    "linknet":           "LinkNet: fast semantic segmentation",
    "pspnet":            "PSPNet: pyramid scene parsing network",
    "resunetpp":         "ResUNet++: residual UNet with SE + ASPP + attention gates",
    "fr_unet":           "FR-UNet: full-resolution UNet with multi-scale fusion",
    "mednext":           "MedNeXt: ConvNeXt-based medical segmentation (2D, size S)",
    # Mamba / SSM
    "h_vmunet":          "H-VMUNet: hybrid vision Mamba UNet",
    "lightm_unet":       "LightM-UNet: lightweight Mamba UNet",
    "swin_umamba":       "Swin-UMamba: Swin Transformer + Mamba UNet",
    "umamba_bot":        "U-Mamba Bot: Mamba at bottleneck (U-Mamba 2024)",
    "umamba_enc":        "U-Mamba Enc: Mamba in encoder (U-Mamba 2024)",
    "ultralight_vmunet": "UltraLight-VMUNet: ultra-lightweight vision Mamba UNet",
    "vm_unet":           "VM-UNet: Vision Mamba UNet",
    "vm_unet_v2":        "VM-UNet V2: Vision Mamba UNet v2",
    "lkm_unet":          "LKM-UNet: large kernel Mamba UNet",
    "log_vmamba":        "LoG-VMamba: local-global Vision Mamba",
    "pvt_mamba":         "PVT-Mamba: pyramid vision transformer + Mamba",
    "vmkla_unet":        "VMKLA-UNet: vision Mamba with kernel-level attention",
    # RWKV
    "u_rwkv":            "U-RWKV: RWKV-based UNet",
    "rwkv_unet":         "RWKV-UNet: RWKV state-space UNet",
    "md_rwkv_unet":      "MD-RWKV-UNet: multi-domain RWKV UNet",
    "rir_zigzag":        "RIR-Zigzag: RWKV-based zigzag UNet",
    # KAN / MLP / LSTM
    "ukan":              "UKAN: U-KAN with Kolmogorov-Arnold networks",
    "xlstm_unet_bot":    "xLSTM-UNet Bot: xLSTM at bottleneck",
    "xlstm_unet_enc":    "xLSTM-UNet Enc: xLSTM in encoder",
    "rolling_unet":      "Rolling-UNet: rolling attention UNet",
    "unext":             "UNeXt: tokenized MLP UNet",
    # Transformer (standalone only, no encoder+decoder YAML yet)
    "double_unet":       "DoubleU-Net: dual encoder-decoder with ASPP",
    "fcbformer":         "FCBFormer: fully convolutional B-transformer",
    "pvt_unet":          "PVT-UNet: pyramid vision transformer UNet",
    "mobile_u_vit":      "MobileU-ViT: mobile-friendly UNet with ViT",
    "cswin_unet":        "CSWin-UNet: cross-stitch window attention UNet",
    "transnetr":         "TransNetR: transformer-based residual network",
    # Other
    "ttt_unet":          "TTT-UNet: UNet with Test-Time Training at bottleneck",
}

# Special arch_params for models that need them
ARCH_PARAMS = {
    "ttt_unet":    {"features": [32, 64, 128, 256, 512], "ttt_d_state": 16},
    "mednext":     {"model_id": "S", "kernel_size": 3},
    "umamba_enc":  {"features": [32, 64, 128, 256, 512]},
}

# Dataset templates
DATASETS = {
    "acdc": {
        "num_classes": 4,
        "img_size": 224,
        "epochs": 200,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 24,
        "data": {
            "type": "acdc",
            "img_size": 224,
            "train_dir": "./data/ACDC/train_npz",
            "test_dir": "./data/ACDC/test_vol_h5",
            "train_list": "./data/ACDC/lists/train.txt",
            "test_list": "./data/ACDC/lists/test_vol.txt",
        },
    },
    "synapse": {
        "num_classes": 9,
        "img_size": 224,
        "epochs": 200,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 24,
        "data": {
            "type": "synapse",
            "img_size": 224,
            "train_dir": "./data/Synapse/train_npz",
            "test_dir": "./data/Synapse/test_vol_h5",
            "train_list": "./data/Synapse/lists/lists_Synapse/train.txt",
            "test_list": "./data/Synapse/lists/lists_Synapse/test_vol.txt",
        },
    },
    "binary": {
        "num_classes": 2,
        "img_size": 256,
        "epochs": 300,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 24,
        "data": {
            "type": "binary",
            "img_size": 256,
            "train_dir": "./data/BinaryDataset/train",
            "val_dir": "./data/BinaryDataset/val",
            "test_dir": "./data/BinaryDataset/test",
        },
    },
}


def _fmt_arch_params(params):
    """Format arch_params dict as YAML string."""
    if not params:
        return "  arch_params: {}\n"
    lines = ["  arch_params:\n"]
    for k, v in params.items():
        if isinstance(v, list):
            lines.append(f"    {k}: [{', '.join(str(x) for x in v)}]\n")
        elif isinstance(v, bool):
            lines.append(f"    {k}: {'true' if v else 'false'}\n")
        elif isinstance(v, (int, float)):
            lines.append(f"    {k}: {v}\n")
        else:
            lines.append(f"    {k}: \"{v}\"\n")
    return "".join(lines)


def generate_yaml(arch_key, description, dataset_name):
    """Generate YAML content for one model + dataset combination."""
    ds = DATASETS[dataset_name]
    arch_params = ARCH_PARAMS.get(arch_key, {})

    lines = []
    # Comment header
    lines.append(f"# {description}\n")
    lines.append(f"# Standalone architecture: {arch_key}\n")
    lines.append(f"# Uses architecture key for direct model dispatch\n")
    lines.append("\n")

    # Model section
    lines.append("model:\n")
    lines.append(f"  num_classes: {ds['num_classes']}\n")
    lines.append(f"  img_size: {ds['img_size']}\n")
    lines.append(f"  architecture: {arch_key}\n")
    lines.append("  encoder:\n")
    lines.append("    in_channels: 3\n")
    lines.append(_fmt_arch_params(arch_params))

    lines.append("\n")

    # Data section
    lines.append("data:\n")
    for k, v in ds["data"].items():
        lines.append(f"  {k}: {v}\n")

    lines.append("\n")

    # Training section
    lines.append("training:\n")
    lines.append(f"  epochs: {ds['epochs']}\n")
    lines.append(f"  batch_size: {ds['batch_size']}\n")
    lines.append("  num_workers: 4\n")
    lines.append("  loss:\n")
    lines.append("    name: compound\n")
    lines.append("    params:\n")
    lines.append("      losses:\n")
    lines.append("      - name: ce\n")
    lines.append("        weight: 0.4\n")
    lines.append("      - name: dice\n")
    lines.append("        weight: 0.6\n")
    lines.append("  optimizer:\n")
    lines.append("    name: adamw\n")
    lines.append(f"    lr: {ds['lr']}\n")
    lines.append(f"    weight_decay: {ds['weight_decay']}\n")
    lines.append("  scheduler:\n")
    lines.append("    name: cosine\n")
    lines.append("    min_lr: 1.0e-06\n")

    return "".join(lines)


def main():
    created = 0
    skipped = 0

    for arch_key, description in MODELS.items():
        for dataset_name in ["acdc", "synapse", "binary"]:
            config_dir = os.path.join(ROOT, dataset_name)
            os.makedirs(config_dir, exist_ok=True)

            filename = f"{arch_key}.yaml"
            filepath = os.path.join(config_dir, filename)

            # Skip if already exists
            if os.path.exists(filepath):
                # Check if existing file uses encoder+decoder format (not architecture key)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if "architecture:" in content:
                    print(f"  SKIP (exists+arch): {dataset_name}/{filename}")
                    skipped += 1
                    continue
                # If it's encoder+decoder format, create standalone version with _standalone suffix
                standalone_path = os.path.join(config_dir, f"{arch_key}_standalone.yaml")
                if os.path.exists(standalone_path):
                    print(f"  SKIP (standalone exists): {dataset_name}/{arch_key}_standalone.yaml")
                    skipped += 1
                    continue
                filepath = standalone_path
                print(f"  CREATE (standalong alongside enc-dec): {dataset_name}/{arch_key}_standalone.yaml")
            else:
                print(f"  CREATE: {dataset_name}/{filename}")

            yaml_content = generate_yaml(arch_key, description, dataset_name)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            created += 1

    print(f"\nDone: {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
