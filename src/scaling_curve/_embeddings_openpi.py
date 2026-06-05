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
