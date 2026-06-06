"""OpenPI model embedding extraction support.

This module provides OpenPIEmbeddingExtractor (PyTorch) and
OpenPIEmbeddingExtractorJAX (JAX) for extracting embeddings
from openpi's π₀.₅ and other models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

try:
    import openpi
    from openpi.training import config as _config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    import safetensors.torch
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import flax.nnx as nnx
    OPENPI_JAX_AVAILABLE = True and OPENPI_AVAILABLE
except ImportError:
    OPENPI_JAX_AVAILABLE = False
    nnx = None  # type: ignore


def _require_openpi() -> None:
    """Raise ImportError if openpi is not available."""
    if not OPENPI_AVAILABLE:
        raise ImportError(
            "openpi is required for OpenPIEmbeddingExtractor. "
            "Install with: pip install 'data-evaluator[openpi]'"
        )


def _require_openpi_jax() -> None:
    """Raise ImportError if openpi JAX support is not available."""
    if not OPENPI_JAX_AVAILABLE:
        raise ImportError(
            "openpi with JAX support is required for OpenPIEmbeddingExtractorJAX. "
            "Install with: pip install 'data-evaluator[openpi]' jax orbax"
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


def _flatten_jax(t: jnp.ndarray) -> jnp.ndarray:
    """Reduce any JAX tensor to [N, D] via global average pooling."""
    if t.ndim <= 2:
        return t
    if t.ndim == 3:
        return t.mean(axis=1)  # [B, seq, D] -> [B, D]
    return t.mean(axis=list(range(2, t.ndim)))  # [B, C, H, W] -> [B, C]


class OpenPIEmbeddingExtractorJAX:
    """Extract per-camera embeddings from openpi JAX models.

    This class directly uses JAX models without conversion to PyTorch.

    Args:
        checkpoint_path: Path to openpi checkpoint directory containing
            params/ (JAX format).
        model_type: Type of openpi model. One of "pi0", "pi05", "pi0_fast".
        hook_module: Dotted path to module to hook.
            pi05/pi0 -> "PaliGemma.img" (SigLIP vision encoder)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: _ModelType = "pi05",
        hook_module: str = "PaliGemma.img",
    ):
        _require_openpi_jax()

        checkpoint_path = Path(checkpoint_path)
        params_path = checkpoint_path / "params"

        if not params_path.exists():
            raise FileNotFoundError(
                f"JAX checkpoint not found at {params_path}. "
                "Ensure checkpoint_path points to a JAX-format checkpoint."
            )

        self._hook_module = hook_module
        self._model_type = model_type

        # Load openpi config
        config = _config.get_config(model_type)

        # Create and load the JAX model
        from openpi.model import restore_params
        from openpi.models import pi0

        params = restore_params(str(params_path), restore_type=np.ndarray)
        self.model = pi0.Pi0(config.model, rngs=nnx.Rngs(0))

        # Load params using orbax
        checkpointer = ocp.PyTreeCheckpointer()
        self.model_state = checkpointer.restore(str(params_path))
        self.model.eval()

        # Register forward hook using JAX's interception
        self._hook_outputs: list[jnp.ndarray] = []
        self._register_hook(hook_module)

    def _register_hook(self, module_path: str):
        """Register hook on the specified module using JAX wrapper."""

        def _wrap_module_for_hook(module, original_method):
            """Wrap a module method to capture outputs."""
            def wrapped(*args, **kwargs):
                result = original_method(*args, **kwargs)
                if isinstance(result, jnp.ndarray):
                    self._hook_outputs.append(result)
                return result
            return wrapped

        # Navigate to the target module
        parts = module_path.split(".")
        module = self.model
        for attr in parts[:-1]:
            module = getattr(module, attr)

        # Wrap the __call__ method of the target
        target = getattr(module, parts[-1])
        setattr(module, parts[-1], _wrap_module_for_hook(target, target.__call__))

    def _prepare_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> dict:
        """Convert PyTorch observation to JAX format expected by openpi.

        Args:
            observation: Dictionary from LeRobotDatasetLoader with torch tensors

        Returns:
            Dictionary in openpi JAX format
        """
        # Convert torch tensors to numpy, then to JAX
        jax_obs = {}

        # Handle images - LeRobot format to openpi format
        # LeRobot: (C, H, W) float32 in [0, 1]
        # openpi: (H, W, C) uint8 in [0, 255] or float32 in [-1, 1]

        images = {}
        image_masks = {}

        # Map observation keys to openpi expected keys
        # This is a simplified mapping - may need customization per dataset
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                # Convert from (C, H, W) to (H, W, C)
                if value.dim() == 3:
                    np_value = value.cpu().numpy().transpose(1, 2, 0)
                    # Scale from [0, 1] to [0, 255] uint8
                    if np_value.dtype == np.float32 or np_value.dtype == np.float64:
                        np_value = (np_value * 255).astype(np.uint8)
                    images[key] = jnp.array(np_value)
                    image_masks[key] = jnp.array(True)

        jax_obs["image"] = images
        jax_obs["image_mask"] = image_masks

        # Add dummy state (required by openpi but not used for embedding extraction)
        jax_obs["state"] = jnp.zeros((7,))  # 7-DOF dummy state

        return jax_obs

    def extract_per_camera(
        self,
        observation: dict[str, torch.Tensor],
        camera_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass and return {camera_key: embedding [D]}.

        Args:
            observation: Dictionary with camera keys mapping to torch tensors.
                Expected format from LeRobotDatasetLoader.
            camera_keys: List of camera keys to extract embeddings for.

        Returns:
            Dictionary mapping each camera key to its embedding tensor (torch).
        """
        # Clear previous hook outputs
        self._hook_outputs = []

        # Prepare observation in JAX format
        jax_obs = self._prepare_observation(observation)

        # Create Observation object
        from openpi.models import model as _model

        obs = _model.Observation(
            images=jax_obs["image"],
            image_masks=jax_obs["image_mask"],
            state=jax_obs["state"],
        )

        # Run forward pass using embed_prefix (processes images through vision encoder)
        with jax.numpy_rank_promotion("allow"):
            tokens, input_mask, ar_mask = self.model.embed_prefix(obs)

        # Process hook outputs (captured from vision encoder)
        outputs = [_flatten_jax(o) for o in self._hook_outputs]
        n_cams = len(camera_keys)

        if not outputs:
            # If hook didn't work, fall back to using the tokens directly
            # tokens shape: [B, seq_len, embed_dim]
            # For multiple cameras, we need to split by camera
            if tokens.ndim == 3:
                # Take mean over sequence dimension to get [B, embed_dim]
                embeddings = tokens.mean(axis=1)
                # Convert back to torch
                embeddings = torch.from_numpy(np.array(embeddings))
                return {k: embeddings.squeeze(0) for k in camera_keys}

            raise RuntimeError(
                f"Hook never fired during forward pass. "
                f"Check that hook_module '{self._hook_module}' is correct."
            )

        # Convert JAX arrays to torch tensors
        torch_outputs = [torch.from_numpy(np.array(o)) for o in outputs]

        # Match outputs to camera keys
        if len(torch_outputs) == n_cams:
            return {k: torch_outputs[i].squeeze(0) for i, k in enumerate(camera_keys)}

        if len(torch_outputs) == 1:
            flat = torch_outputs[0]
            if flat.shape[0] == n_cams:
                return {k: flat[i] for i, k in enumerate(camera_keys)}
            emb = flat.squeeze(0)
            return {k: emb for k in camera_keys}

        raise RuntimeError(
            f"Hook fired {len(outputs)} time(s) but expected {n_cams} camera(s). "
            "Try a different hook_module path."
        )
