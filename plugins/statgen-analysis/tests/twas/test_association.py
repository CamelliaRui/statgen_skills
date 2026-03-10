# tests/twas/test_association.py
"""Tests for TWAS association testing."""

import pytest
import numpy as np


def test_compute_twas_z():
    """Test TWAS Z-score computation."""
    from scripts.twas.association import compute_twas_z

    np.random.seed(42)
    n_samples = 1000

    # Simulated predicted expression and phenotype
    pred_expression = np.random.randn(n_samples)

    # Correlated phenotype (should give significant Z)
    phenotype = 0.5 * pred_expression + np.random.randn(n_samples) * 0.5

    z, p = compute_twas_z(pred_expression, phenotype)

    assert isinstance(z, float)
    assert isinstance(p, float)
    assert p < 0.05  # Should be significant


def test_run_twas_returns_results():
    """Test running TWAS on multiple genes."""
    from scripts.twas.association import run_twas

    np.random.seed(42)
    n_samples = 500
    n_genes = 10

    # Simulated predicted expression matrix
    pred_expression = np.random.randn(n_samples, n_genes)

    # Phenotype correlated with first 3 genes
    true_effects = np.zeros(n_genes)
    true_effects[:3] = [0.3, 0.2, 0.1]
    phenotype = pred_expression @ true_effects + np.random.randn(n_samples) * 0.5

    results = run_twas(pred_expression, phenotype)

    assert "z_scores" in results
    assert "p_values" in results
    assert len(results["z_scores"]) == n_genes

    # First genes should have lower p-values
    assert results["p_values"][0] < results["p_values"][-1]


def test_compute_power_fdr():
    """Test power and FDR computation."""
    from scripts.twas.association import compute_power_fdr

    n_genes = 100
    n_causal = 10

    # Mock results
    p_values = np.random.uniform(0, 1, n_genes)
    # Make causal genes significant
    p_values[:n_causal] = np.random.uniform(0, 0.01, n_causal)

    causal_mask = np.zeros(n_genes, dtype=bool)
    causal_mask[:n_causal] = True

    metrics = compute_power_fdr(p_values, causal_mask, alpha=0.05)

    assert "power" in metrics
    assert "fdr" in metrics
    assert "n_discoveries" in metrics
    assert 0 <= metrics["power"] <= 1
    assert 0 <= metrics["fdr"] <= 1
