# tests/fusion/test_fusion_plots.py
"""Tests for FUSION visualization functions."""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_create_fusion_locus_plot():
    """Test regional locus plot creation."""
    from visualization.fusion_plots import create_fusion_locus_plot

    # Mock TWAS results
    twas_df = pd.DataFrame({
        "gene": ["GENE1", "GENE2", "GENE3"],
        "chromosome": [1, 1, 1],
        "start": [1000000, 1500000, 2000000],
        "end": [1100000, 1600000, 2100000],
        "twas_z": [4.5, -2.1, 3.2],
        "twas_p": [1e-5, 0.03, 0.001],
    })

    # Mock GWAS results
    gwas_df = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(100)],
        "BP": np.linspace(500000, 2500000, 100),
        "P": np.random.uniform(0.001, 1, 100),
    })
    gwas_df.loc[:5, "P"] = np.random.uniform(1e-8, 1e-5, 6)

    fig = create_fusion_locus_plot(
        twas_df=twas_df,
        gwas_df=gwas_df,
        chromosome=1,
        region_start=500000,
        region_end=2500000,
    )

    assert fig is not None
    plt.close(fig)


def test_create_tissue_heatmap():
    """Test multi-tissue heatmap creation."""
    from visualization.fusion_plots import create_tissue_heatmap

    # Mock multi-tissue results
    results = pd.DataFrame({
        "gene": ["GENE1", "GENE2", "GENE3"] * 3,
        "tissue": ["Whole_Blood"] * 3 + ["Brain_Cortex"] * 3 + ["Liver"] * 3,
        "twas_z": [4.5, 2.1, 3.2, 3.8, 1.5, 2.9, 2.2, 0.5, 1.8],
    })

    fig = create_tissue_heatmap(results)
    assert fig is not None
    plt.close(fig)
