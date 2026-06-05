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


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """Reduce any tensor to [N, D] via global average pooling."""
    if t.dim() <= 2:
        return t
    if t.dim() == 3:
        return t.mean(dim=1)  # [B, seq, D] -> [B, D]
    return t.mean(dim=list(range(2, t.dim())))  # [B, C, H, W] -> [B, C]


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

    def extract_per_camera(
        self,
        observation: dict[str, torch.Tensor],
        camera_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass and return {camera_key: embedding [D]}.

        Args:
            observation: Dictionary with camera keys mapping to image tensors.
                Expected format from LeRobotDatasetLoader.
            camera_keys: List of camera keys to extract embeddings for.

        Returns:
            Dictionary mapping each camera key to its embedding tensor.
        """
        # Prepare batch: add batch dimension and move to device
        batch = {
            k: v.unsqueeze(0).to(self.device)
            for k, v in observation.items()
            if isinstance(v, torch.Tensor)
        }

        # Clear previous hook outputs
        self._hook_outputs = []

        # Run forward pass
        with torch.no_grad():
            # Note: openpi models don't have select_action, we need to call
            # the model's forward method directly
            # For now, this is a placeholder - the actual forward call
            # will depend on how openpi's PI0Pytorch model is used
            if hasattr(self.model, "sample_actions"):
                # Try the sample_actions method if available
                _ = self.model.sample_actions(batch)
            else:
                # Otherwise we need to call forward directly
                # This will be implemented based on openpi's actual API
                raise NotImplementedError(
                    "Direct forward pass not yet implemented. "
                    "OpenPI model requires specific input formatting."
                )

        # Process hook outputs
        outputs = [_flatten(o) for o in self._hook_outputs]
        n_cams = len(camera_keys)

        if not outputs:
            raise RuntimeError(
                "Hook never fired during forward pass. "
                f"Check that hook_module '{self._hook_module}' is correct."
            )

        # Match outputs to camera keys
        if len(outputs) == n_cams:
            return {k: outputs[i].squeeze(0) for i, k in enumerate(camera_keys)}

        if len(outputs) == 1:
            flat = outputs[0]
            if flat.shape[0] == n_cams:
                return {k: flat[i] for i, k in enumerate(camera_keys)}
            emb = flat.squeeze(0)
            return {k: emb for k in camera_keys}

        raise RuntimeError(
            f"Hook fired {len(outputs)} time(s) but expected {n_cams} camera(s). "
            "Try a different hook_module path."
        )
