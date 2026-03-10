# visualization/twas_plots.py
"""
TWAS visualization functions.

Creates publication-ready figures for:
- Power curves across parameter values
- Model comparison plots
- TWAS Manhattan plots
- QQ plots for p-value calibration
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_power_curve(
    results: pd.DataFrame,
    vary_param: str,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    color: str = "#3498db",
) -> plt.Figure:
    """
    Create a power curve showing power vs parameter value.
    
    Args:
        results: DataFrame with columns: param_value, power, power_se (optional)
        vary_param: Name of the varied parameter (for axis label)
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        color: Line color
        
    Returns:
        matplotlib Figure
    """
    results = results.copy()
    results.columns = results.columns.str.lower()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = results["param_value"]
    y = results["power"]
    
    # Plot with error bands if SE provided
    if "power_se" in results.columns:
        se = results["power_se"]
        ax.fill_between(x, y - 1.96*se, y + 1.96*se, alpha=0.2, color=color)
    
    ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel(vary_param, fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_ylim(0, 1)
    
    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(f"TWAS Power vs {vary_param}", fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_model_comparison(
    results: pd.DataFrame,
    metric: str = "cv_r2",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Create a bar plot comparing model performance.
    
    Args:
        results: DataFrame with columns: model, <metric>, <metric>_se (optional)
        metric: Which metric to plot
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    results = results.copy()
    results.columns = results.columns.str.lower()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(results))
    colors = sns.color_palette("husl", len(results))
    
    # Plot bars with error bars if SE available
    se_col = f"{metric}_se"
    if se_col in results.columns:
        ax.bar(x, results[metric], yerr=results[se_col] * 1.96,
               color=colors, capsize=5, edgecolor="white", linewidth=0.5)
    else:
        ax.bar(x, results[metric], color=colors, edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(results["model"], rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_twas_manhattan(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6),
    title: str = "TWAS Manhattan Plot",
    significance_threshold: float = 5e-8,
    suggestive_threshold: float = 1e-5,
) -> plt.Figure:
    """
    Create a Manhattan plot for TWAS results.
    
    Args:
        results: DataFrame with columns: gene, chromosome, position, p_value
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        significance_threshold: Genome-wide significance
        suggestive_threshold: Suggestive significance
        
    Returns:
        matplotlib Figure
    """
    results = results.copy()
    results.columns = results.columns.str.lower()
    
    # Calculate -log10(p)
    results["log_p"] = -np.log10(results["p_value"].clip(1e-300))
    
    # Sort by chromosome and position
    results = results.sort_values(["chromosome", "position"])
    
    # Create cumulative position
    chrom_offsets = {}
    offset = 0
    for chrom in sorted(results["chromosome"].unique()):
        chrom_offsets[chrom] = offset
        offset += results[results["chromosome"] == chrom]["position"].max() + 1e7
    
    results["cumulative_pos"] = results.apply(
        lambda r: r["position"] + chrom_offsets[r["chromosome"]], axis=1
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color alternating chromosomes
    colors = ["#2ecc71", "#3498db"]
    for i, chrom in enumerate(sorted(results["chromosome"].unique())):
        chrom_data = results[results["chromosome"] == chrom]
        ax.scatter(
            chrom_data["cumulative_pos"],
            chrom_data["log_p"],
            c=colors[i % 2],
            s=20,
            alpha=0.7,
        )
    
    # Add significance lines
    ax.axhline(-np.log10(significance_threshold), color="red", linestyle="--",
               linewidth=1, label=f"p = {significance_threshold}")
    ax.axhline(-np.log10(suggestive_threshold), color="blue", linestyle=":",
               linewidth=1, label=f"p = {suggestive_threshold}")
    
    # Chromosome labels
    chrom_centers = results.groupby("chromosome")["cumulative_pos"].median()
    ax.set_xticks(chrom_centers.values)
    ax.set_xticklabels(chrom_centers.index.astype(int))
    
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_qq_plot(
    p_values: np.ndarray,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (7, 7),
    title: str = "QQ Plot",
) -> plt.Figure:
    """
    Create a QQ plot for p-value calibration.
    
    Args:
        p_values: Array of p-values
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    # Remove zeros and clip very small values
    p_values = np.array(p_values)
    p_values = p_values[p_values > 0]
    p_values = np.clip(p_values, 1e-300, 1)
    
    # Sort and compute expected
    observed = -np.log10(np.sort(p_values))
    n = len(p_values)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Identity line
    max_val = max(observed.max(), expected.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Expected")
    
    # QQ points
    ax.scatter(expected, observed, c="#3498db", s=15, alpha=0.6)
    
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(title, fontweight="bold")
    
    # Compute lambda (genomic inflation)
    lambda_gc = np.median(observed) / 0.455
    ax.text(0.05, 0.95, f"λ = {lambda_gc:.2f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig
