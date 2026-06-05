# OpenPI Pi0.5 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for openpi's π₀.₅ model to enable scaling curve evaluation with Physical Intelligence's VLA model.

**Architecture:** Create an independent `OpenPIEmbeddingExtractor` class that loads openpi PyTorch models, uses forward hooks to extract embeddings from the vision encoder, and integrates with the existing scaling curve generation pipeline.

**Tech Stack:** Python 3.11+, PyTorch, openpi (PyTorch), safetensors, LeRobot datasets

---

## File Structure

**New files:**
- `src/scaling_curve/_embeddings_openpi.py` - OpenPI embedding extractor implementation
- `src/scaling_curve/tests/test_embeddings_openpi.py` - Unit tests for openpi extractor

**Modified files:**
- `pyproject.toml` - Add openpi optional dependency
- `src/scaling_curve/__init__.py` - Export new classes
- `src/scaling_curve/scaling_curve.py` - Integrate openpi support

---

## Task 1: Add Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add openpi optional dependency**

```toml
[project.optional-dependencies]
openpi = ["openpi"]
```

- [ ] **Step 2: Verify change**

Run: `cat pyproject.toml`
Expected: See `[project.optional-dependencies]` section with `openpi = ["openpi"]`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add openpi as optional dependency"
```

---

## Task 2: Create OpenPI Embedding Extractor Skeleton

**Files:**
- Create: `src/scaling_curve/_embeddings_openpi.py`

- [ ] **Step 1: Create file with imports and availability flag**

```python
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
```

- [ ] **Step 2: Verify file created**

Run: `ls -la src/scaling_curve/_embeddings_openpi.py`
Expected: File exists with above content

- [ ] **Step 3: Commit**

```bash
git add src/scaling_curve/_embeddings_openpi.py
git commit -m "feat: add openpi extractor skeleton with imports"
```

---

## Task 3: Implement OpenPIEmbeddingExtractor.__init__

**Files:**
- Modify: `src/scaling_curve/_embeddings_openpi.py`

- [ ] **Step 1: Add OpenPIEmbeddingExtractor class with __init__ method**

Add after `_ModelType` definition:

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/scaling_curve/_embeddings_openpi.py`
Expected: No syntax errors

- [ ] **Step 3: Commit**

```bash
git add src/scaling_curve/_embeddings_openpi.py
git commit -m "feat: implement OpenPIEmbeddingExtractor.__init__"
```

---

## Task 4: Implement extract_per_camera Method

**Files:**
- Modify: `src/scaling_curve/_embeddings_openpi.py`

- [ ] **Step 1: Add _flatten helper and extract_per_camera method**

Add before `OpenPIEmbeddingExtractor` class:

```python
def _flatten(t: torch.Tensor) -> torch.Tensor:
    """Reduce any tensor to [N, D] via global average pooling."""
    if t.dim() <= 2:
        return t
    if t.dim() == 3:
        return t.mean(dim=1)  # [B, seq, D] -> [B, D]
    return t.mean(dim=list(range(2, t.dim())))  # [B, C, H, W] -> [B, C]
```

Add `extract_per_camera` method to `OpenPIEmbeddingExtractor` class (after `__del__`):

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/scaling_curve/_embeddings_openpi.py`
Expected: No syntax errors

- [ ] **Step 3: Commit**

```bash
git add src/scaling_curve/_embeddings_openpi.py
git commit -m "feat: implement extract_per_camera method"
```

---

## Task 5: Add Tests for OpenPIEmbeddingExtractor

**Files:**
- Create: `src/scaling_curve/tests/test_embeddings_openpi.py`

- [ ] **Step 1: Create test file with basic structure**

```python
"""Tests for openpi embedding extractor."""

import pytest
import torch

from scaling_curve._embeddings_openpi import (
    OpenPIEmbeddingExtractor,
    OPENPI_AVAILABLE,
    _require_openpi,
    _flatten,
)


@pytest.mark.skipif(not OPENPI_AVAILABLE, reason="openpi not installed")
class TestOpenPIEmbeddingExtractor:
    """Test OpenPIEmbeddingExtractor functionality."""

    def test_require_openpi_when_available(self):
        """Test that _require_openpi doesn't raise when openpi is available."""
        # Should not raise
        _require_openpi()

    def test_flatten_2d(self):
        """Test _flatten with 2D tensor (no-op)."""
        t = torch.randn(4, 128)
        result = _flatten(t)
        assert result.shape == (4, 128)

    def test_flatten_3d(self):
        """Test _flatten with 3D tensor (sequence pooling)."""
        t = torch.randn(4, 10, 128)
        result = _flatten(t)
        assert result.shape == (4, 128)

    def test_flatten_4d(self):
        """Test _flatten with 4D tensor (spatial pooling)."""
        t = torch.randn(4, 64, 8, 8)
        result = _flatten(t)
        assert result.shape == (4, 64)

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create a minimal mock checkpoint structure."""
        # This would create a minimal checkpoint for testing
        # For now, skip actual model loading tests
        pass

    def test_init_requires_checkpoint(self):
        """Test that __init__ requires a valid checkpoint."""
        with pytest.raises(FileNotFoundError):
            OpenPIEmbeddingExtractor(
                checkpoint_path="/nonexistent/path",
                model_type="pi05",
            )


class TestOpenPIAvailability:
    """Test openpi availability checks."""

    def test_require_openpi_when_unavailable(self, monkeypatch):
        """Test that _require_openpi raises when openpi is not available."""
        # Mock OPENPI_AVAILABLE to False
        import scaling_curve._embeddings_openpi as emb_module
        monkeypatch.setattr(emb_module, "OPENPI_AVAILABLE", False)

        with pytest.raises(ImportError, match="openpi is required"):
            _require_openpi()
```

- [ ] **Step 2: Verify file created**

Run: `ls -la src/scaling_curve/tests/test_embeddings_openpi.py`
Expected: File exists with above content

- [ ] **Step 3: Run tests**

Run: `pytest src/scaling_curve/tests/test_embeddings_openpi.py -v`
Expected: Tests that don't require openpi should pass; openpi tests should be skipped

- [ ] **Step 4: Commit**

```bash
git add src/scaling_curve/tests/test_embeddings_openpi.py
git commit -m "test: add basic tests for OpenPIEmbeddingExtractor"
```

---

## Task 6: Update __init__.py to Export OpenPI Classes

**Files:**
- Modify: `src/scaling_curve/__init__.py`

- [ ] **Step 1: Add openpi exports**

Current content likely looks like:
```python
from scaling_curve._dataset import LeRobotDatasetLoader
from scaling_curve._embeddings import PolicyEmbeddingExtractor
from scaling_curve._similarity import policy_embedding_similarity
from scaling_curve.scaling_curve import (
    ScalingCurveGenerator,
    MultiScalingCurveGenerator,
)

__all__ = [
    "LeRobotDatasetLoader",
    "PolicyEmbeddingExtractor",
    "policy_embedding_similarity",
    "ScalingCurveGenerator",
    "MultiScalingCurveGenerator",
]
```

Add to imports and exports:
```python
from scaling_curve._dataset import LeRobotDatasetLoader
from scaling_curve._embeddings import PolicyEmbeddingExtractor
from scaling_curve._embeddings_openpi import OpenPIEmbeddingExtractor
from scaling_curve._similarity import policy_embedding_similarity
from scaling_curve.scaling_curve import (
    ScalingCurveGenerator,
    MultiScalingCurveGenerator,
)

__all__ = [
    "LeRobotDatasetLoader",
    "PolicyEmbeddingExtractor",
    "OpenPIEmbeddingExtractor",
    "policy_embedding_similarity",
    "ScalingCurveGenerator",
    "MultiScalingCurveGenerator",
]
```

- [ ] **Step 2: Verify exports work**

Run: `python -c "from scaling_curve import OpenPIEmbeddingExtractor; print('OK')"`
Expected: Prints "OK" (even if openpi not installed, the import should work)

- [ ] **Step 3: Commit**

```bash
git add src/scaling_curve/__init__.py
git commit -m "feat: export OpenPIEmbeddingExtractor from package"
```

---

## Task 7: Update ScalingCurveGenerator Documentation

**Files:**
- Modify: `src/scaling_curve/scaling_curve.py`

- [ ] **Step 1: Add note about openpi support to docstring**

Find the `ScalingCurveGenerator` class docstring and add to the "Limitations" section:

```python
    Limitations:
        ...
        - **Supported policies**: ACT, DiffusionPolicy, Pi0, Pi05, Pi0Fast, TDMPC, VQBeT
          (lerobot>=0.4.0). For openpi models, use OpenPIEmbeddingExtractor directly.
```

- [ ] **Step 2: Verify docstring**

Run: `python -c "from scaling_curve import ScalingCurveGenerator; help(ScalingCurveGenerator.__init__)"`
Expected: See updated docstring

- [ ] **Step 3: Commit**

```bash
git add src/scaling_curve/scaling_curve.py
git commit -m "docs: note openpi support in ScalingCurveGenerator docstring"
```

---

## Task 8: Create Usage Example Documentation

**Files:**
- Create: `docs/openpi_usage.md`

- [ ] **Step 1: Create usage documentation**

```markdown
# OpenPI Model Support

This document describes how to use openpi models with the scaling curve evaluator.

## Installation

Install the package with openpi support:

```bash
pip install 'data-evaluator[openpi]'
# or
pip install openpi
```

## Basic Usage

```python
from scaling_curve import OpenPIEmbeddingExtractor
from scaling_curve._dataset import LeRobotDatasetLoader

# Initialize the extractor
extractor = OpenPIEmbeddingExtractor(
    checkpoint_path="path/to/pi05_checkpoint",
    model_type="pi05",
    hook_module="paligemma_with_expert.paligemma.vision_tower",
    device="cuda",
)

# Load your data
train_loader = LeRobotDatasetLoader("data/train")
eval_loader = LeRobotDatasetLoader("data/eval")

train_obs = train_loader.get_initial_observations()
eval_obs = eval_loader.get_initial_observations()

# Extract embeddings
for obs in train_obs:
    embeddings = extractor.extract_per_camera(obs, ["front"])
    # embeddings["front"] is the embedding tensor
```

## Supported Model Types

- `pi0`: Original π₀ model
- `pi05`: π₀.₅ model with knowledge insulation
- `pi0_fast`: Autoregressive FAST version (PyTorch support coming soon)

## Hook Module Options

Different hook points for embedding extraction:

| Hook Module | Description |
|-------------|-------------|
| `paligemma_with_expert.paligemma.vision_tower` | SigLIP vision encoder output |
| `paligemma_with_expert.gemma_expert.model` | Action expert features |

## Requirements

- openpi PyTorch checkpoint (converted from JAX if needed)
- LeRobot v3.0 format dataset
- CUDA GPU recommended for inference
```

- [ ] **Step 2: Verify file created**

Run: `ls -la docs/openpi_usage.md`
Expected: File exists with above content

- [ ] **Step 3: Commit**

```bash
git add docs/openpi_usage.md
git commit -m "docs: add openpi usage documentation"
```

---

## Task 9: Update README with OpenPI Support

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add openpi to supported models section**

Find the section that lists supported models and add:

```markdown
### Supported Policy Models

- **LeRobot policies**: ACT, DiffusionPolicy, Pi0, Pi0Fast, TDMPC, VQBeT
- **openpi policies**: π₀, π₀.₅ (requires `pip install 'data-evaluator[openpi]'`)

See [docs/openpi_usage.md](docs/openpi_usage.md) for openpi usage.
```

- [ ] **Step 2: Verify README still valid**

Run: `cat README.md | head -50`
Expected: See updated supported models section

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add openpi to supported models in README"
```

---

## Task 10: Integration Testing (Manual)

**Files:**
- None (manual verification)

- [ ] **Step 1: Test import without openpi**

Run: `python -c "from scaling_curve import OpenPIEmbeddingExtractor; print('Import OK')"`
Expected: "Import OK" printed

- [ ] **Step 2: Verify error message without openpi**

Run: `python -c "from scaling_curve import OpenPIEmbeddingExtractor; e = OpenPIEmbeddingExtractor('/tmp', 'pi05')"`
Expected: ImportError with helpful message about installing openpi

- [ ] **Step 3: Document test completion**

This step is complete when:
- Import test passes without openpi
- ImportError message is clear and helpful
- No exceptions or crashes

No git commit needed for manual verification.

---

## Verification Checklist

After completing all tasks:

- [ ] All tests pass: `pytest src/scaling_curve/tests/ -v`
- [ ] Import works without openpi installed
- [ ] Import works with openpi installed
- [ ] Documentation is complete and accurate
- [ ] README reflects new capabilities
- [ ] Git history shows clean, incremental commits
