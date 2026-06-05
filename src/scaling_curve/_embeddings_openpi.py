"""OpenPI model embedding extraction support.

This module provides OpenPIEmbeddingExtractor for extracting embeddings
from openpi's π₀.₅ and other models using PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

try:
    import openpi
    from openpi.training import config as _config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False


def _require_openpi() -> None:
    """Raise ImportError if openpi is not available."""
    if not OPENPI_AVAILABLE:
        raise ImportError(
            "openpi is required for OpenPIEmbeddingExtractor. "
            "Install with: pip install 'data-evaluator[openpi]'"
        )


_ModelType = Literal["pi0", "pi05", "pi0_fast"]


class OpenPIEmbeddingExtractor:
    """Extract per-camera embeddings from openpi models via forward hook.

    Args:
        checkpoint_path: Path to openpi checkpoint directory containing
            model.safetensors (PyTorch format).
        model_type: Type of openpi model. One of "pi0", "pi05", "pi0_fast".
        hook_module: Dotted path to module to hook.
            pi05/pi0 -> "paligemma_with_expert.paligemma.vision_tower"
        device: "auto", "cpu", "cuda", or "cuda:N".
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: _ModelType = "pi05",
        hook_module: str = "paligemma_with_expert.paligemma.vision_tower",
        device: str = "auto",
    ):
        _require_openpi()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self._hook_module = hook_module

        # Load openpi config
        config = _config.get_config(model_type)

        # Create PyTorch model
        self.model = PI0Pytorch(config.model)
        self.model.eval()

        # Load weights from checkpoint
        checkpoint_path = Path(checkpoint_path)
        weight_path = checkpoint_path / "model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {weight_path}. "
                "Ensure checkpoint_path points to a PyTorch-format checkpoint."
            )
        safetensors.torch.load_model(self.model, str(weight_path))
        self.model.to(self.device)

        # Register forward hook
        self._hook_outputs: list[torch.Tensor] = []
        self._hook_handle = self._register_hook(hook_module)

    def _register_hook(self, module_path: str):
        """Register forward hook on the specified module."""
        module = self.model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        def _hook(mod, inp, output):
            if isinstance(output, torch.Tensor):
                self._hook_outputs.append(output.detach().cpu())
            elif isinstance(output, dict):
                vals = [v for v in output.values() if isinstance(v, torch.Tensor)]
                if vals:
                    self._hook_outputs.append(vals[-1].detach().cpu())
            elif isinstance(output, (tuple, list)) and output:
                if isinstance(output[0], torch.Tensor):
                    self._hook_outputs.append(output[0].detach().cpu())

        return module.register_forward_hook(_hook)

    def __del__(self):
        """Clean up hook on deletion."""
        if hasattr(self, "_hook_handle") and self._hook_handle:
            self._hook_handle.remove()
