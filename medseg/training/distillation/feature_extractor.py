"""Feature extractor helper for hook-based intermediate feature capture.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class FeatureExtractor(nn.Module):
    """Extract intermediate features from UNet models."""

    def __init__(self, model, feature_layers: Optional[List[str]] = None):
        super().__init__()
        self.model = model
        self.feature_layers = feature_layers or []
        self.features = {}

        # Register hooks for feature extraction
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features.

        Supports two match modes:
        1. Exact match: ``name == layer_name``
        2. Suffix match: ``name`` ends with ``layer_name`` or with an
           additional ``.model.`` infix (e.g. ``encoder.model.layer3``
           matches the config entry ``encoder.layer3`` for timm encoders
           that wrap their backbone under ``.model``).

        Only the shallowest matching module is hooked to avoid redundant
        captures from every sub-layer of the same block.
        """
        def make_hook(layer_name):
            def hook(module, input, output):
                # If the module returns a sequence (e.g. encoder returning
                # multiple feature maps), take the last (deepest) one as the
                # representative feature for KD losses.
                if isinstance(output, (list, tuple)):
                    output = output[-1]
                self.features[layer_name] = output
            return hook

        def _matches(name: str, layer_name: str) -> bool:
            if name == layer_name:
                return True
            # Allow timm .model. infix: "encoder.layer3" -> "encoder.model.layer3"
            parts = layer_name.split('.')
            for insert_pos in range(1, len(parts)):
                candidate = '.'.join(parts[:insert_pos]) + '.model.' + '.'.join(parts[insert_pos:])
                if name == candidate:
                    return True
            return False

        for layer_name in self.feature_layers:
            for name, module in self.model.named_modules():
                if _matches(name, layer_name):
                    module.register_forward_hook(make_hook(layer_name))
                    break  # hook only the first (shallowest) match

    def forward(self, x):
        """Forward pass and return both output and features."""
        output = self.model(x)
        return output, self.features.copy()
