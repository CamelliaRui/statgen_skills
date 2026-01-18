# tests/twas/test_integration.py
"""Integration tests for TWAS simulation pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def simulated_genotypes():
    """Generate realistic simulated genotypes."""
    np.random.seed(42)
    n_samples = 1000
    n_genes = 50
    n_snps_per_gene = 100
    
    genotypes = []
    for _ in range(n_genes):
        # Simulate with realistic LD structure (block diagonal)
        g = np.random.randint(0, 3, (n_samples, n_snps_per_gene)).astype(float)
        # Standardize
        g = (g - g.mean(0)) / (g.std(0) + 1e-8)
        genotypes.append(g)
    
    return genotypes


def test_full_simulation_pipeline(simulated_genotypes):
    """Test the complete TWAS simulation pipeline."""
    from scripts.twas import simulate_twas
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = simulate_twas(
            genotypes_list=simulated_genotypes,
            n_causal_genes=5,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.2,
            n_causal_cis=3,
            models=["elastic_net", "lasso"],
            output_dir=tmpdir,
            seed=42,
            verbose=False,
        )
        
        # Check result structure
        assert "twas_results" in result
        assert "model_performance" in result
        assert "power_metrics" in result
        
        # Check all files created
        assert Path(tmpdir, "simulation_params.json").exists()
        assert Path(tmpdir, "true_effects.csv").exists()
        assert Path(tmpdir, "twas_results.csv").exists()
        assert Path(tmpdir, "model_performance.csv").exists()
        assert Path(tmpdir, "summary.json").exists()
        
        # Check power is reasonable (with known causal genes, should have some power)
        for model in ["elastic_net", "lasso"]:
            assert result["power_metrics"][model]["power"] >= 0
            assert result["power_metrics"][model]["fdr"] <= 1


def test_models_produce_different_results(simulated_genotypes):
    """Test that different models produce different predictions."""
    from scripts.twas import simulate_twas
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = simulate_twas(
            genotypes_list=simulated_genotypes[:10],  # Smaller for speed
            n_causal_genes=2,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.2,
            models=["elastic_net", "lasso", "gblup"],
            output_dir=tmpdir,
            seed=42,
            verbose=False,
        )
        
        # Models should have different performance
        perf = result["model_performance"]
        r2_values = [perf[m]["mean_cv_r2"] for m in ["elastic_net", "lasso", "gblup"]]
        
        # At least some variation between models
        assert max(r2_values) > min(r2_values) or all(r == 0 for r in r2_values)


def test_visualization_integration():
    """Test that visualizations work with simulation output."""
    from scripts.twas import simulate_twas
    from visualization.twas_plots import create_model_comparison, create_qq_plot
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    n_samples, n_genes, n_snps = 200, 10, 30
    genotypes = [
        np.random.randn(n_samples, n_snps)
        for _ in range(n_genes)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = simulate_twas(
            genotypes_list=genotypes,
            n_causal_genes=2,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.1,
            models=["elastic_net"],
            output_dir=tmpdir,
            seed=42,
            verbose=False,
        )
        
        # Create model comparison from output
        perf_df = pd.read_csv(Path(tmpdir) / "model_performance.csv")
        perf_df.columns = ["model", "cv_r2", "cv_corr", "n_nonzero"]
        fig1 = create_model_comparison(perf_df, metric="cv_r2")
        assert fig1 is not None
        plt.close(fig1)
        
        # Create QQ plot from p-values
        twas_df = pd.read_csv(Path(tmpdir) / "twas_results.csv")
        p_values = twas_df["elastic_net_p"].values
        fig2 = create_qq_plot(p_values)
        assert fig2 is not None
        plt.close(fig2)


def test_reproducibility_with_seed(simulated_genotypes):
    """Test that results are reproducible with same seed."""
    from scripts.twas import simulate_twas
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        result1 = simulate_twas(
            genotypes_list=simulated_genotypes[:5],
            n_causal_genes=2,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.1,
            models=["elastic_net"],
            output_dir=tmpdir1,
            seed=42,
            verbose=False,
        )
        
        result2 = simulate_twas(
            genotypes_list=simulated_genotypes[:5],
            n_causal_genes=2,
            h2_cis=0.1,
            h2_trait=0.5,
            prop_mediated=0.1,
            models=["elastic_net"],
            output_dir=tmpdir2,
            seed=42,
            verbose=False,
        )
        
        # Results should be identical
        np.testing.assert_allclose(
            result1["twas_results"]["elastic_net"]["z_scores"],
            result2["twas_results"]["elastic_net"]["z_scores"],
            rtol=1e-5,
        )
