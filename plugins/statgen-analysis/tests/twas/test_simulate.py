# tests/twas/test_simulate.py
"""Tests for main TWAS simulation."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def mock_genotypes():
    """Generate mock genotype data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_genes = 20
    n_snps_per_gene = 50

    genotypes = []
    for _ in range(n_genes):
        g = np.random.randint(0, 3, (n_samples, n_snps_per_gene)).astype(float)
        # Standardize
        g = (g - g.mean(0)) / (g.std(0) + 1e-8)
        genotypes.append(g)

    return genotypes


def test_simulate_twas_basic(mock_genotypes):
    """Test basic TWAS simulation."""
    from scripts.twas.simulate import simulate_twas

    with tempfile.TemporaryDirectory() as tmpdir:
        result = simulate_twas(
            genotypes_list=mock_genotypes,
            n_causal_genes=5,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.2,
            models=["elastic_net"],
            output_dir=tmpdir,
            seed=42,
        )

        assert "twas_results" in result
        assert "model_performance" in result
        assert "true_effects" in result
        assert "power_metrics" in result

        # Check output files created
        assert Path(tmpdir, "simulation_params.json").exists()
        assert Path(tmpdir, "twas_results.csv").exists()


def test_simulate_twas_multiple_models(mock_genotypes):
    """Test simulation with multiple models."""
    from scripts.twas.simulate import simulate_twas

    with tempfile.TemporaryDirectory() as tmpdir:
        result = simulate_twas(
            genotypes_list=mock_genotypes,
            n_causal_genes=3,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.1,
            models=["elastic_net", "lasso"],
            output_dir=tmpdir,
            seed=42,
        )

        # Should have results for each model
        assert "elastic_net" in result["twas_results"]
        assert "lasso" in result["twas_results"]

        # Performance metrics for each model
        assert "elastic_net" in result["model_performance"]
        assert "lasso" in result["model_performance"]


def test_simulate_twas_saves_outputs(mock_genotypes):
    """Test that all expected output files are saved."""
    from scripts.twas.simulate import simulate_twas

    with tempfile.TemporaryDirectory() as tmpdir:
        simulate_twas(
            genotypes_list=mock_genotypes,
            n_causal_genes=3,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.1,
            models=["elastic_net"],
            output_dir=tmpdir,
            seed=42,
        )

        # Check all expected files
        expected_files = [
            "simulation_params.json",
            "true_effects.csv",
            "twas_results.csv",
            "model_performance.csv",
            "summary.json",
        ]

        for fname in expected_files:
            assert Path(tmpdir, fname).exists(), f"Missing: {fname}"
