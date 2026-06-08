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
    from openpi.models import pi0_config as _pi0_config
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


def _get_model_config(model_type: _ModelType):
    """Return the model config object for the given model type."""
    if model_type == "pi05":
        return _pi0_config.Pi0Config(action_horizon=15, pi05=True)
    if model_type == "pi0_fast":
        from openpi.models import pi0_fast as _pi0_fast
        return _pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)
    return _pi0_config.Pi0Config()


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
        model_config = _get_model_config(model_type)

        # Create PyTorch model
        self.model = PI0Pytorch(model_config)
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
        # Clear previous hook outputs
        self._hook_outputs = []

        # Transform LeRobot format to openpi format
        # LeRobot: {camera_key: tensor[C, H, W]}
        # openpi expects images as dict of tensors

        images = {}
        image_masks = {}

        for key in camera_keys:
            if key in observation:
                img = observation[key]
                # LeRobot images are [C, H, W] in [0, 1], convert to what openpi expects
                # openpi expects [B, H, W, C] normalized to [-1, 1] or [0, 255]
                if img.dim() == 3:
                    img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                    # Normalize from [0, 1] to [-1, 1] (expected by SigLIP)
                    img = img * 2.0 - 1.0
                    images[key] = img.unsqueeze(0).to(self.device)  # Add batch dim
                    image_masks[key] = torch.tensor([True], device=self.device)

        # Create dummy state (required but not used for vision embedding)
        state = torch.zeros((1, 7), device=self.device)  # 7-DOF dummy state

        # Create observation object (using a simple namespace-like object)
        class Observation:
            def __init__(self, images, image_masks, state):
                self.images = images
                self.image_masks = image_masks
                self.state = state
                self.tokenized_prompt = None
                self.tokenized_prompt_mask = None

        obs = Observation(images, image_masks, state)

        # Run forward pass using embed_prefix to trigger vision encoder
        with torch.no_grad():
            try:
                # Call embed_prefix which processes images through vision encoder
                # This will trigger the hook on vision_tower
                images_list = list(obs.images.values())
                masks_list = list(obs.image_masks.values())

                _ = self.model.embed_prefix(
                    images=images_list,
                    img_masks=masks_list,
                    lang_tokens=None,
                    lang_masks=None
                )
            except Exception as e:
                # Fallback: try direct method
                raise RuntimeError(
                    f"Failed to run openpi model: {e}. "
                    f"Check that input format is compatible with {self._model_type}."
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
        model_config = _get_model_config(model_type)

        # Create and load the JAX model
        from openpi.models import model as _model

        params = _model.restore_params(params_path, restore_type=np.ndarray)
        self.model = model_config.load(params)
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

    def extract_per_camera(
        self,
        observation: dict[str, torch.Tensor],
        camera_keys: list[str],
    ) -> dict[str, torch.Tensor]:
        """Run the SigLIP vision encoder per camera and return {camera_key: embedding [D]}.

        Directly calls self.model.PaliGemma.img for each image, bypassing
        the unreliable NNX hook mechanism.

        Args:
            observation: Dictionary with camera keys mapping to torch tensors
                in LeRobot format: {key: torch(C,H,W) float32 [0,1]}.
            camera_keys: List of camera keys to extract embeddings for.

        Returns:
            Dictionary mapping each camera key to its embedding tensor [D] (torch float32).
        """
        from openpi_client import image_tools

        result = {}
        for key in camera_keys:
            if key not in observation:
                continue
            img = observation[key]
            # [C,H,W] float [0,1] → [H,W,C] uint8 [0,255]
            np_img = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            # resize to 224×224 with padding, add batch dim → [1,224,224,3]
            np_img = image_tools.resize_with_pad(np_img[None], 224, 224)
            # uint8 → float32 [-1, 1]
            jax_img = jnp.array(np_img, dtype=jnp.float32) / 255.0 * 2.0 - 1.0

            # image_tokens: [1, num_patches, embed_dim]  (num_patches=256 for So400m/14)
            image_tokens, _ = self.model.PaliGemma.img(jax_img, train=False)
            print(f"[img encoder] key={key} tokens shape={image_tokens.shape} "
                  f"dtype={image_tokens.dtype} mean={float(image_tokens.mean()):.4f} "
                  f"std={float(image_tokens.std()):.4f}")

            # global average pool over patches → [1, embed_dim] → [embed_dim]
            emb = image_tokens.mean(axis=1).squeeze(0)
            result[key] = torch.from_numpy(np.array(emb, dtype=np.float32))

        return result
