# OpenPI Model Support

This document describes how to use openpi models with the scaling curve evaluator.

## Installation

Install the package with openpi support:

```bash
pip install 'data-evaluator[openpi]'
# or
pip install openpi
```

For JAX support, additionally install:
```bash
pip install jax orbax
```

## PyTorch vs JAX

This project provides two extractors for openpi models:

| Extractor | Checkpoint Format | Pros | Cons |
|-----------|------------------|------|------|
| `OpenPIEmbeddingExtractor` | PyTorch (`model.safetensors`) | Faster inference, wider compatibility | Requires conversion from JAX |
| `OpenPIEmbeddingExtractorJAX` | JAX (`params/` directory) | No conversion needed, works with original weights | Requires JAX installation |

**Note:** Original openpi checkpoints are in JAX format. You can either:
1. Use `OpenPIEmbeddingExtractorJAX` directly with JAX checkpoints
2. Convert JAX to PyTorch and use `OpenPIEmbeddingExtractor` (see Conversion section below)

## PyTorch Usage

```python
from scaling_curve import OpenPIEmbeddingExtractor

# Requires PyTorch-format checkpoint (model.safetensors)
extractor = OpenPIEmbeddingExtractor(
    checkpoint_path="path/to/pi05_pytorch_checkpoint",
    model_type="pi05",
    hook_module="paligemma_with_expert.paligemma.vision_tower",
    device="cuda",
)
```

## JAX Usage

```python
from scaling_curve import OpenPIEmbeddingExtractorJAX

# Works directly with JAX checkpoints (params/ directory)
extractor = OpenPIEmbeddingExtractorJAX(
    checkpoint_path="path/to/pi05_jax_checkpoint",
    model_type="pi05",
    hook_module="PaliGemma.img",
)
```

## Complete Example (JAX)

```python
from scaling_curve import OpenPIEmbeddingExtractorJAX
from scaling_curve._dataset import LeRobotDatasetLoader

# Initialize the JAX extractor
extractor = OpenPIEmbeddingExtractorJAX(
    checkpoint_path="~/.cache/openpi/openpi-assets/checkpoints/pi05_droid",
    model_type="pi05",
)

# Load your data
train_loader = LeRobotDatasetLoader("data/train")
eval_loader = LeRobotDatasetLoader("data/eval")

train_obs = train_loader.get_initial_observations()
eval_obs = eval_loader.get_initial_observations()

# Extract embeddings
for obs in train_obs:
    embeddings = extractor.extract_per_camera(obs, train_loader.camera_keys)
    # embeddings[camera_key] is the embedding tensor
```

## Supported Model Types

- `pi0`: Original π₀ model
- `pi05`: π₀.₅ model with knowledge insulation
- `pi0_fast`: Autoregressive FAST version

## JAX to PyTorch Conversion

If you want to use `OpenPIEmbeddingExtractor` (PyTorch), convert JAX checkpoints first:

```bash
# Clone openpi repository
git clone git@github.com:Physical-Intelligence/openpi.git
cd openpi

# Install openpi
pip install -e .

# Convert checkpoint
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --output_path ~/.cache/openpi/pi05_droid_pytorch \
    --config_name pi05_droid \
    --precision bfloat16
```

## Hook Module Options

| Extractor | Hook Module | Description |
|----------|-------------|-------------|
| PyTorch | `paligemma_with_expert.paligemma.vision_tower` | SigLIP vision encoder output |
| PyTorch | `paligemma_with_expert.gemma_expert.model` | Action expert features |
| JAX | `PaliGemma.img` | SigLIP vision encoder |

## Requirements

**PyTorch:**
- openpi PyTorch checkpoint (converted from JAX)
- LeRobot v3.0 format dataset
- CUDA GPU recommended

**JAX:**
- openpi JAX checkpoint (original format)
- JAX and orbax installed
- LeRobot v3.0 format dataset
- CUDA GPU recommended
