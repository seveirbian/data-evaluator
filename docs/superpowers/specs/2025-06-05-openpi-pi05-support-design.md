# OpenPI Pi0.5 Support Design

**Date**: 2025-06-05
**Author**: Claude Code + User Collaboration
**Status**: Approved

## Overview

Add support for openpi's π₀.₅ model to the scaling curve evaluation project. openpi is an independent robotics VLA model project by Physical Intelligence, separate from lerobot.

## Problem Statement

Currently, the project only supports lerobot-based policies through `_POLICY_REGISTRY`. openpi provides π₀.₅ (pi0.5), an upgraded version of π₀ with better open-world generalization, which would be valuable to support.

## Key Findings

### openpi Architecture
- **Framework**: JAX/Flax (primary), with PyTorch support
- **PyTorch Model**: `PI0Pytorch` class in `openpi.models_pytorch.pi0_pytorch`
- **Vision Encoder**: SigLIP via `paligemma_with_expert.paligemma.vision_tower`
- **Checkpoint Format**: safetensors (PyTorch version)
- **LeRobot Support**: openpi has built-in LeRobot dataset support via transforms

### Current Project Architecture
- **Loading**: `_POLICY_REGISTRY` mapping to lerobot policies
- **Extraction**: `PolicyEmbeddingExtractor` with forward hooks
- **Data**: LeRobot v3.0 format via `LeRobotDatasetLoader`

## Design Decision: Independent Extractor (Option A)

Chose to create a separate `OpenPIEmbeddingExtractor` class rather than extending the existing lerobot registry.

**Rationale**:
1. openpi has fundamentally different loading mechanisms
2. Clear separation of concerns
3. Easier to maintain and debug
4. Reusable pattern for future non-lerobot models

## Architecture

```
src/scaling_curve/
├── _embeddings.py           # Existing: lerobot support
├── _embeddings_openpi.py    # New: openpi support
├── scaling_curve.py         # Updated: integrate openpi
└── ...
```

## Core Components

### 1. OpenPIEmbeddingExtractor Class

**File**: `src/scaling_curve/_embeddings_openpi.py`

```python
class OpenPIEmbeddingExtractor:
    """Extract per-camera embeddings from openpi models.

    Args:
        checkpoint_path: Path to openpi checkpoint directory
        model_type: "pi05", "pi0", "pi0_fast", etc.
        hook_module: Dotted path to module for hook
            pi05/pi0 -> "paligemma_with_expert.paligemma.vision_tower"
        device: "auto", "cpu", "cuda", or "cuda:N"
        camera_key_map: LeRobot camera keys → openpi camera names
    """

    def __init__(self, checkpoint_path: str, model_type: str,
                 hook_module: str, device: str = "auto",
                 camera_key_map: dict[str, str] | None = None):
        ...

    def extract_per_camera(
        self, observation: dict[str, torch.Tensor],
        camera_keys: list[str]
    ) -> dict[str, torch.Tensor]:
        """Run forward pass and return {camera_key: embedding [D]}."""
```

### 2. Camera Key Mapping

Generic mapping interface to support any robot platform:

```python
# Example usage for 星海图R1 robot
gen = ScalingCurveGenerator(
    checkpoint_path="path/to/pi05_checkpoint",
    train_data_dir="星海图R1/train",
    eval_data_dir="星海图R1/eval",
    camera_key_map={
        "camera_1": "base_0_rgb",
        "camera_2": "left_wrist_0_rgb",
        "camera_3": "right_wrist_0_rgb",
    }
)
```

### 3. Dependency Management

openpi as optional dependency:

**pyproject.toml**:
```toml
[project.optional-dependencies]
openpi = ["openpi"]
```

**Import handling**:
```python
try:
    import openpi
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False

if not OPENPI_AVAILABLE:
    raise ImportError(
        "openpi is required for OpenPIEmbeddingExtractor. "
        "Install with: pip install openpi"
    )
```

## Model Loading Flow

```
1. Load openpi config
   config = _config.get_config(model_type)

2. Create PyTorch model
   model = PI0Pytorch(config.model)

3. Load weights
   safetensors.torch.load_model(model, checkpoint_path)

4. Register forward hook
   module = get_module_by_path(hook_module)
   hook = module.register_forward_hook(...)
```

## Hook Targets

| Hook Module Path | Component | Output |
|-----------------|-----------|--------|
| `paligemma_with_expert.paligemma.vision_tower` | SigLIP vision encoder | Visual embeddings |
| `paligemma_with_expert.gemma_expert.model` | Action expert | Action embeddings |

## Data Flow

```
LeRobot Dataset
    ↓
LeRobotDatasetLoader (existing)
    ↓
Observation dict {camera_key: tensor}
    ↓
Camera key mapping (if configured)
    ↓
OpenPI transforms (via openpi Policy system)
    ↓
PI0Pytorch forward pass
    ↓
Hook captures intermediate output
    ↓
Flatten to [N, D]
    ↓
{camera_key: embedding} dict
```

## Implementation Files

1. **src/scaling_curve/_embeddings_openpi.py** (new)
   - `OpenPIEmbeddingExtractor` class
   - Model loading logic
   - Hook registration
   - Data processing

2. **src/scaling_curve/scaling_curve.py** (update)
   - Import openpi extractor
   - Add conditional logic to choose extractor

3. **pyproject.toml** (update)
   - Add openpi optional dependency

## Limitations

1. **Checkpoint format**: Only supports PyTorch-format checkpoints (not JAX)
2. **openpi installation**: Users must install openpi separately
3. **Camera requirements**: openpi expects certain camera configurations

## Testing Plan

1. **Unit tests**: Mock openpi model, test hook mechanism
2. **Integration tests**: Test with real pi05 checkpoint
3. **End-to-end**: Generate scaling curve with pi05

## Future Considerations

- Support JAX-format checkpoints if needed
- Add support for pi0_fast if PyTorch version becomes available
- Consider unified factory pattern for additional non-lerobot models
