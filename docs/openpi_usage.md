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
