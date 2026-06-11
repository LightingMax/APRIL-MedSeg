"""Text-guided segmentation decoder modules.

These models are **text-guided decoders** that pair with an external visual
encoder (ViT, ResNet, etc.) built by ``train_text_guided.py``.  The training
script extracts multi-scale features from the encoder and feeds them here.

For **end-to-end** text-guided models (CRIS, TGANet, LViT, BiomedParse, etc.)
that have their own visual + text backbone, see ``medseg/models/text_unet/``.

Two decoder architectures are provided:

1. **TextPromptUNet** — UNet decoder with text cross-attention + FiLM
   modulation at every decoder stage.  Text embeddings come from a CLIP
   text encoder (HF transformers) or learnable prompts (CoOp-style).

2. **SemanticGuidedUNet** — lighter decoder that injects learnable class
   embeddings via channel-attention gating at skip connections.

Contract with ``train_text_guided.py``:
    * ``forward(encoder_features: List[Tensor]) -> Tensor``
    * ``encoder_features`` is ordered ``[shallow, ..., deep]`` (same order
      as ``encoder.out_channels``).
    * Returns segmentation logits ``(B, num_classes, H, W)`` at the input
      image resolution (2× the shallowest feature map).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Text encoders
# =====================================================================

class CLIPTextEncoder(nn.Module):
    """Encode class-name phrases into a ``(num_classes, embed_dim)`` matrix.

    Requires HuggingFace ``transformers`` (``CLIPTextModel`` + ``CLIPTokenizer``).
    If the dependency is missing or the model cannot be downloaded, an error
    is raised immediately — there is **no silent fallback** to learnable
    embeddings.  Use :class:`LearnableTextPrompts` explicitly when CLIP is
    not available.

    Args:
        class_names: one natural-language phrase per segmentation class.
        model_name: HuggingFace CLIP model id.
        embed_dim: desired output embedding width.  A linear projection is
            inserted when the native CLIP hidden size differs.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
    ):
        super().__init__()
        self.class_names = class_names or ["background", "foreground"]

        # HF transformers is a hard requirement — no try/except
        from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore

        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._clip_text_model = CLIPTextModel.from_pretrained(model_name)
        clip_dim = self._clip_text_model.config.hidden_size  # 512 for ViT-B/32

        # Freeze CLIP text tower weights
        for p in self._clip_text_model.parameters():
            p.requires_grad = False

        self._projection: Optional[nn.Linear] = None
        if clip_dim != embed_dim:
            self._projection = nn.Linear(clip_dim, embed_dim)
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self) -> torch.Tensor:
        """Return ``(num_classes, embed_dim)`` text embeddings."""
        tokens = self._tokenizer(
            self.class_names,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            out = self._clip_text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        feats = out.pooler_output                          # (K, clip_dim)
        if self._projection is not None:
            feats = self._projection(feats)
        return feats                                       # (K, embed_dim)


class LearnableTextPrompts(nn.Module):
    """CoOp-style learnable prompt vectors, one set per class.

    Args:
        num_classes: number of segmentation classes.
        embed_dim: output embedding width.
        num_tokens: learnable context token count (default 16).
    """

    def __init__(self, num_classes: int, embed_dim: int = 512, num_tokens: int = 16):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(num_classes, num_tokens, embed_dim) * 0.02)
        # Class-specific bias so each class has a distinct anchor
        self.cls_bias = nn.Embedding(num_classes, embed_dim)
        nn.init.trunc_normal_(self.cls_bias.weight, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self.ctx.shape[-1]

    def forward(self) -> torch.Tensor:
        """Return ``(num_classes, embed_dim)``."""
        bias = self.cls_bias.weight.unsqueeze(1)          # (K, 1, D)
        prompt = torch.cat([self.ctx, bias], dim=1)        # (K, T+1, D)
        return prompt.mean(dim=1)                           # (K, D)


# =====================================================================
# Text ↔ vision fusion primitives
# =====================================================================

class _TextCrossAttention(nn.Module):
    """Visual features attend to text embeddings (multi-head).

    Operates on 4-D visual tensors ``(B, C, H, W)`` and a 2-D text matrix
    ``(K, D)`` where *K* is the number of classes.
    """

    def __init__(self, vis_dim: int, txt_dim: int, num_heads: int = 8):
        super().__init__()
        assert vis_dim % num_heads == 0, "vis_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = vis_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_conv = nn.Conv2d(vis_dim, vis_dim, 1)
        self.k_lin = nn.Linear(txt_dim, vis_dim)
        self.v_lin = nn.Linear(txt_dim, vis_dim)
        self.out_conv = nn.Conv2d(vis_dim, vis_dim, 1)
        self.norm = nn.GroupNorm(8, vis_dim)

    def forward(self, vis: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = vis.shape
        K = txt.shape[0]
        nh, hd = self.num_heads, self.head_dim

        q = self.q_conv(vis).view(B, nh, hd, H * W).permute(0, 1, 3, 2)   # (B, nh, HW, hd)
        k = self.k_lin(txt).view(K, nh, hd).permute(1, 0, 2)              # (nh, K, hd)
        v = self.v_lin(txt).view(K, nh, hd).permute(1, 0, 2)              # (nh, K, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale                      # (B, nh, HW, K)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)         # (B, C, H, W)
        out = self.out_conv(out)
        return self.norm(vis + out)                                        # residual


class _FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation conditioned on text.

    ``out = vis * (1 + scale(text)) + shift(text)``
    """

    def __init__(self, vis_dim: int, txt_dim: int):
        super().__init__()
        self.scale_fc = nn.Linear(txt_dim, vis_dim)
        self.shift_fc = nn.Linear(txt_dim, vis_dim)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, vis: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        t = txt.mean(dim=0)                                    # (D,)
        s = self.scale_fc(t).view(1, -1, 1, 1)                 # (1, C, 1, 1)
        b = self.shift_fc(t).view(1, -1, 1, 1)
        return vis * (1 + s) + b


# =====================================================================
# Decoder building blocks
# =====================================================================

class _DecoderBlock(nn.Module):
    """Single UNet decoder stage: upsample → concat skip → conv → text fuse.

    Args:
        in_ch: channels of the incoming (deeper) feature.
        skip_ch: channels of the skip (shallower) feature.
        txt_dim: text embedding width.
        use_text_attn: apply cross-attention after conv.
        use_text_film: apply FiLM modulation after cross-attention.
    """

    def __init__(
        self, in_ch: int, skip_ch: int, txt_dim: int,
        use_text_attn: bool = True, use_text_film: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_ch * 2, skip_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, skip_ch),
            nn.GELU(),
            nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, skip_ch),
            nn.GELU(),
        )
        self.text_attn = (
            _TextCrossAttention(skip_ch, txt_dim) if use_text_attn else nn.Identity()
        )
        self.text_film = (
            _FiLMModulation(skip_ch, txt_dim) if use_text_film else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, txt: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial mismatch (encoder with odd input sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.text_attn(x, txt)
        x = self.text_film(x, txt)
        return x


# =====================================================================
# TextPromptUNet — main text-guided decoder
# =====================================================================

class TextPromptUNet(nn.Module):
    """UNet decoder with text cross-attention + FiLM at every stage.

    The model expects multi-scale features from an external encoder ordered
    ``[shallow, …, deep]`` (e.g. ``[64, 128, 256, 512]`` channels).  It
    decodes from the deepest feature back to the shallowest, fusing text
    embeddings at each stage via:

        1. cross-attention  (visual tokens attend to class text embeddings)
        2. FiLM modulation  (global text → per-channel scale & shift)

    A final 1×1 head produces per-pixel logits.

    Args:
        num_classes: number of segmentation classes.
        class_names: natural-language name per class (for CLIP encoding).
        encoder_channels: channel widths of the external encoder, shallow→deep.
        prompt_mode: ``'clip'`` uses :class:`CLIPTextEncoder`,
            ``'learnable'`` uses :class:`LearnableTextPrompts`.
        embed_dim: text embedding width.
    """

    def __init__(
        self,
        num_classes: int = 9,
        class_names: Optional[List[str]] = None,
        encoder_channels: Optional[List[int]] = None,
        prompt_mode: str = "learnable",
        embed_dim: int = 512,
        img_size: int = 224,
    ):
        super().__init__()
        if encoder_channels is None:
            raise ValueError(
                "encoder_channels is required — pass the external encoder's "
                "out_channels list (e.g. [128, 256, 512, 1024])."
            )
        _VALID_MODES = ("clip", "learnable")
        if prompt_mode not in _VALID_MODES:
            raise ValueError(
                f"prompt_mode must be one of {_VALID_MODES}, got '{prompt_mode}'"
            )

        self.num_classes = num_classes
        self.encoder_channels = list(encoder_channels)

        # ---- text encoder ----
        if prompt_mode == "clip":
            self.text_encoder = CLIPTextEncoder(
                class_names=class_names, embed_dim=embed_dim
            )
        else:
            self.text_encoder = LearnableTextPrompts(
                num_classes=num_classes, embed_dim=embed_dim
            )
        txt_dim = self.text_encoder.embed_dim

        # ---- decoder stages (deep → shallow) ----
        self.decoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1, 0, -1):
            self.decoders.append(
                _DecoderBlock(
                    in_ch=encoder_channels[i],
                    skip_ch=encoder_channels[i - 1],
                    txt_dim=txt_dim,
                )
            )

        # ---- segmentation head ----
        # Final upsample 2× to match input resolution (encoder stride-2 stem)
        self.head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(8, encoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], num_classes, 1),
        )

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features with text guidance.

        Args:
            encoder_features: ``[f_shallow, …, f_deep]``, each ``(B, C_i, H_i, W_i)``.

        Returns:
            Logits ``(B, num_classes, 2*H_0, 2*W_0)`` where ``H_0, W_0``
            are the spatial dims of the shallowest feature.
        """
        txt = self.text_encoder()                       # (K, D)

        x = encoder_features[-1]                        # deepest
        for i, dec in enumerate(self.decoders):
            skip = encoder_features[-(i + 2)]
            x = dec(x, skip, txt)

        # Upsample 2× to recover the stride-2 stem resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


# =====================================================================
# SemanticGuidedUNet — lighter alternative
# =====================================================================

class _ChannelGate(nn.Module):
    """Squeeze-and-excitation gate conditioned on class embeddings."""

    def __init__(self, ch: int, txt_dim: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(txt_dim + ch, mid),
            nn.GELU(),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, vis: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """vis: (B, C, H, W)  txt: (D,)"""
        pooled = vis.mean(dim=[2, 3])                           # (B, C)
        t = txt.unsqueeze(0).expand(pooled.shape[0], -1)        # (B, D)
        gate = self.fc(torch.cat([pooled, t], dim=1))           # (B, C)
        return vis * gate.view(-1, vis.shape[1], 1, 1)


class SemanticGuidedUNet(nn.Module):
    """Lighter text-guided decoder using channel-attention gating.

    Instead of full cross-attention at every stage, this model uses
    squeeze-and-excitation gates conditioned on learnable class embeddings.
    Good for scenarios where compute budget is limited.

    Args:
        num_classes: number of segmentation classes.
        encoder_channels: channel widths of the external encoder, shallow→deep.
        embed_dim: class embedding width.
    """

    def __init__(
        self,
        num_classes: int = 9,
        encoder_channels: Optional[List[int]] = None,
        embed_dim: int = 256,
    ):
        super().__init__()
        if encoder_channels is None:
            raise ValueError(
                "encoder_channels is required — pass the external encoder's "
                "out_channels list (e.g. [128, 256, 512, 1024])."
            )

        self.num_classes = num_classes
        self.encoder_channels = list(encoder_channels)

        # Per-class learnable embeddings
        self.class_embeds = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )

        # Global text vector = mean of class embeds (used for gating)
        # Decoder stages
        self.decoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1, 0, -1):
            in_ch = encoder_channels[i]
            skip_ch = encoder_channels[i - 1]
            self.decoders.append(nn.ModuleDict({
                "up": nn.ConvTranspose2d(in_ch, skip_ch, 2, 2),
                "conv": nn.Sequential(
                    nn.Conv2d(skip_ch * 2, skip_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(8, skip_ch),
                    nn.GELU(),
                    nn.Conv2d(skip_ch, skip_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(8, skip_ch),
                    nn.GELU(),
                ),
                "gate": _ChannelGate(skip_ch, embed_dim),
            }))

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(8, encoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], num_classes, 1),
        )

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        txt_global = self.class_embeds.mean(dim=0)       # (D,)

        x = encoder_features[-1]
        for i, dec in enumerate(self.decoders):
            skip = encoder_features[-(i + 2)]
            x = dec["up"](x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec["conv"](x)
            x = dec["gate"](x, txt_global)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)
