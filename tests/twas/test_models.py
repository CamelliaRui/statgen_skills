# tests/twas/test_models.py
"""Tests for expression prediction models."""

import pytest
import numpy as np


def test_model_registry_returns_available_models():
    """Test model registry lists available models."""
    from scripts.twas.models import get_available_models

    models = get_available_models()
    assert "elastic_net" in models
    assert "lasso" in models
    assert "gblup" in models
    assert "true_effects" in models


def test_get_model_raises_for_unknown():
    """Test error for unknown model name."""
    from scripts.twas.models import get_model

    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


def test_base_model_interface():
    """Test that base model defines required interface."""
    from scripts.twas.models.base import ExpressionModel

    # Check required methods exist
    assert hasattr(ExpressionModel, "fit")
    assert hasattr(ExpressionModel, "predict")
    assert hasattr(ExpressionModel, "get_weights")
    assert hasattr(ExpressionModel, "get_nonzero_snps")
    assert hasattr(ExpressionModel, "cross_validate")
