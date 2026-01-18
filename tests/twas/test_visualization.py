# tests/twas/test_visualization.py
"""Tests for TWAS visualization functions."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_create_power_curve():
    """Test power curve plotting."""
    from visualization.twas_plots import create_power_curve
    
    results = pd.DataFrame({
        "param_value": [100, 250, 500, 1000],
        "power": [0.2, 0.4, 0.6, 0.8],
        "power_se": [0.05, 0.04, 0.03, 0.02],
    })
    
    fig = create_power_curve(results, vary_param="n_eqtl")
    assert fig is not None
    plt.close(fig)


def test_create_model_comparison():
    """Test model comparison barplot."""
    from visualization.twas_plots import create_model_comparison
    
    results = pd.DataFrame({
        "model": ["elastic_net", "lasso", "gblup"],
        "cv_r2": [0.15, 0.12, 0.18],
        "cv_r2_se": [0.02, 0.02, 0.03],
    })
    
    fig = create_model_comparison(results)
    assert fig is not None
    plt.close(fig)


def test_create_twas_manhattan():
    """Test TWAS Manhattan plot."""
    from visualization.twas_plots import create_twas_manhattan
    
    np.random.seed(42)
    results = pd.DataFrame({
        "gene": [f"GENE{i}" for i in range(100)],
        "chromosome": np.random.choice(range(1, 23), 100),
        "position": np.random.randint(1e6, 1e8, 100),
        "z_score": np.random.randn(100) * 2,
        "p_value": np.random.uniform(0, 1, 100),
    })
    # Make some significant
    results.loc[:5, "p_value"] = np.random.uniform(1e-10, 1e-5, 6)
    
    fig = create_twas_manhattan(results)
    assert fig is not None
    plt.close(fig)


def test_create_qq_plot():
    """Test QQ plot for p-values."""
    from visualization.twas_plots import create_qq_plot
    
    np.random.seed(42)
    p_values = np.random.uniform(0, 1, 1000)
    # Add some significant ones
    p_values[:10] = np.random.uniform(1e-8, 1e-4, 10)
    
    fig = create_qq_plot(p_values)
    assert fig is not None
    plt.close(fig)
