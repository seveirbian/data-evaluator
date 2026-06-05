"""Tests for openpi embedding extractor."""

import pytest
import torch

from scaling_curve._embeddings_openpi import (
    OpenPIEmbeddingExtractor,
    OPENPI_AVAILABLE,
    _require_openpi,
    _flatten,
)


class TestFlatten:
    """Test _flatten helper function (doesn't require openpi)."""

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


@pytest.mark.skipif(not OPENPI_AVAILABLE, reason="openpi not installed")
class TestOpenPIEmbeddingExtractor:
    """Test OpenPIEmbeddingExtractor functionality."""

    def test_require_openpi_when_available(self):
        """Test that _require_openpi doesn't raise when openpi is available."""
        # Should not raise
        _require_openpi()

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
