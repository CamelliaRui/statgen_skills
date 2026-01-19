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


def test_elastic_net_fit_and_predict():
    """Test ElasticNet model fitting and prediction."""
    from scripts.twas.models import get_model

    np.random.seed(42)
    n_samples, n_snps = 100, 50
    X = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)

    # True sparse effects
    true_weights = np.zeros(n_snps)
    true_weights[:5] = np.random.randn(5) * 0.5
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    model = get_model("elastic_net")
    model.fit(X, y)

    # Check predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # Check weights are sparse
    weights = model.get_weights()
    assert weights.shape == (n_snps,)
    assert len(model.get_nonzero_snps()) < n_snps  # Should be sparse


def test_lasso_fit_and_predict():
    """Test LASSO model fitting and prediction."""
    from scripts.twas.models import get_model

    np.random.seed(42)
    n_samples, n_snps = 100, 50
    X = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)
    true_weights = np.zeros(n_snps)
    true_weights[:3] = [0.5, -0.3, 0.2]
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    model = get_model("lasso")
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # LASSO should be very sparse
    n_nonzero = len(model.get_nonzero_snps())
    assert n_nonzero < n_snps


def test_cross_validate_returns_metrics():
    """Test cross-validation returns proper metrics."""
    from scripts.twas.models import get_model

    np.random.seed(42)
    n_samples, n_snps = 200, 30
    X = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)
    true_weights = np.zeros(n_snps)
    true_weights[:5] = np.random.randn(5)
    y = X @ true_weights + np.random.randn(n_samples) * 0.5

    model = get_model("elastic_net")
    metrics = model.cross_validate(X, y, cv=5, seed=42)

    assert "cv_r2" in metrics
    assert "cv_corr" in metrics
    assert "n_nonzero" in metrics
    assert 0 <= metrics["cv_r2"] <= 1
    assert -1 <= metrics["cv_corr"] <= 1


def test_gblup_fit_and_predict():
    """Test GBLUP model fitting and prediction."""
    from scripts.twas.models import get_model

    np.random.seed(42)
    n_samples, n_snps = 100, 50
    X = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)

    # Dense effects (GBLUP assumes many small effects)
    true_weights = np.random.randn(n_snps) * 0.1
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    model = get_model("gblup")
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # GBLUP uses all SNPs (not sparse)
    weights = model.get_weights()
    assert weights.shape == (n_snps,)
    # Most weights should be non-zero
    assert len(model.get_nonzero_snps()) > n_snps * 0.5


def test_true_effects_perfect_prediction():
    """Test true effects model with known weights."""
    from scripts.twas.models import get_model

    np.random.seed(42)
    n_samples, n_snps = 100, 50
    X = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)

    # Known true weights
    true_weights = np.zeros(n_snps)
    true_weights[:5] = [1.0, -0.5, 0.3, -0.2, 0.1]
    y_true = X @ true_weights

    # Initialize with true weights
    model = get_model("true_effects", true_weights=true_weights)
    model.fit(X, y_true)  # fit is a no-op for true effects

    y_pred = model.predict(X)

    # Should be perfect prediction
    np.testing.assert_allclose(y_pred, y_true, rtol=1e-10)

    # Weights should match
    np.testing.assert_allclose(model.get_weights(), true_weights)
