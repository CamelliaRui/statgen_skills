# tests/visualization/test_ldsc_plots.py
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


def test_create_h2_barplot_returns_figure():
    from visualization.ldsc_plots import create_h2_barplot
    import matplotlib.pyplot as plt

    results = pd.DataFrame({
        "trait": ["Height", "BMI", "T2D"],
        "h2": [0.45, 0.25, 0.15],
        "h2_se": [0.05, 0.03, 0.02],
    })

    fig = create_h2_barplot(results)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_rg_heatmap_returns_figure():
    from visualization.ldsc_plots import create_rg_heatmap
    import matplotlib.pyplot as plt

    # Correlation matrix
    rg_matrix = pd.DataFrame(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]],
        index=["T1", "T2", "T3"],
        columns=["T1", "T2", "T3"],
    )

    fig = create_rg_heatmap(rg_matrix)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_enrichment_plot_returns_figure():
    from visualization.ldsc_plots import create_enrichment_plot
    import matplotlib.pyplot as plt

    results = pd.DataFrame({
        "category": ["Enhancer", "Promoter", "Coding"],
        "enrichment": [2.5, 1.8, 3.2],
        "enrichment_se": [0.3, 0.2, 0.4],
        "enrichment_p": [1e-5, 0.001, 1e-8],
    })

    fig = create_enrichment_plot(results)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
