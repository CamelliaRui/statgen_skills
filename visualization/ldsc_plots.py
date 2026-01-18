# visualization/ldsc_plots.py
"""
LDSC visualization functions.

Creates publication-ready figures for:
- Heritability bar charts
- Genetic correlation heatmaps
- s-LDSC enrichment plots
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_h2_barplot(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 6),
    title: str = "SNP Heritability Estimates",
    color: str = "#3498db",
) -> plt.Figure:
    """
    Create a bar plot of heritability estimates with error bars.

    Args:
        results: DataFrame with columns: trait, h2, h2_se
        output_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title
        color: Bar color

    Returns:
        matplotlib Figure
    """
    results = results.copy()
    results.columns = results.columns.str.lower()

    required = ["trait", "h2", "h2_se"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by h2 descending
    results = results.sort_values("h2", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(results))

    ax.barh(
        y_pos,
        results["h2"],
        xerr=results["h2_se"] * 1.96,  # 95% CI
        color=color,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["trait"])
    ax.set_xlabel("SNP Heritability (h²)")
    ax.set_xlim(0, min(1.0, results["h2"].max() * 1.5))
    ax.set_title(title, fontweight="bold")

    # Add value labels
    for i, (h2, se) in enumerate(zip(results["h2"], results["h2_se"])):
        ax.text(
            h2 + se * 1.96 + 0.02,
            i,
            f"{h2:.3f}",
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved h2 plot to: {output_path}")

    return fig


def create_rg_heatmap(
    rg_matrix: pd.DataFrame,
    pvalue_matrix: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "Genetic Correlations",
    cmap: str = "RdBu_r",
    annot: bool = True,
) -> plt.Figure:
    """
    Create a heatmap of genetic correlations.

    Args:
        rg_matrix: Square DataFrame of genetic correlations
        pvalue_matrix: Optional p-values for significance annotation
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        cmap: Colormap name
        annot: Annotate cells with values

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle (including diagonal)
    mask = np.triu(np.ones_like(rg_matrix, dtype=bool))

    # Create annotation with significance stars
    if annot:
        if pvalue_matrix is not None:
            annot_matrix = rg_matrix.copy().astype(str)
            for i in range(len(rg_matrix)):
                for j in range(len(rg_matrix)):
                    val = rg_matrix.iloc[i, j]
                    pval = pvalue_matrix.iloc[i, j]
                    stars = ""
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                    annot_matrix.iloc[i, j] = f"{val:.2f}{stars}"
            annot_data = annot_matrix
        else:
            annot_data = True
    else:
        annot_data = False

    sns.heatmap(
        rg_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot_data if isinstance(annot_data, bool) else annot_data.values,
        fmt="" if isinstance(annot_data, pd.DataFrame) else ".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Genetic Correlation (rg)"},
        ax=ax,
    )

    ax.set_title(title, fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved rg heatmap to: {output_path}")

    return fig


def create_enrichment_plot(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Heritability Enrichment by Annotation",
    significance_threshold: float = 0.05,
) -> plt.Figure:
    """
    Create a forest plot of s-LDSC enrichment results.

    Args:
        results: DataFrame with columns: category, enrichment, enrichment_se, enrichment_p
        output_path: Path to save figure
        figsize: Figure size
        title: Plot title
        significance_threshold: P-value threshold for highlighting

    Returns:
        matplotlib Figure
    """
    results = results.copy()
    results.columns = results.columns.str.lower()

    required = ["category", "enrichment", "enrichment_se"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by enrichment
    results = results.sort_values("enrichment", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(results))

    # Plot each point and error bar individually to allow different colors
    for i, (_, row) in enumerate(results.iterrows()):
        # Determine color based on significance
        if "enrichment_p" in results.columns:
            if row["enrichment_p"] < significance_threshold:
                color = "#e74c3c" if row["enrichment"] > 1 else "#3498db"
            else:
                color = "#95a5a6"
        else:
            color = "#3498db"

        # Plot error bar
        ax.errorbar(
            row["enrichment"],
            i,
            xerr=row["enrichment_se"] * 1.96,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            capthick=2,
            markersize=8,
        )

    # Add reference line at enrichment = 1
    ax.axvline(x=1, color="black", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["category"])
    ax.set_xlabel("Enrichment (fold)")
    ax.set_title(title, fontweight="bold")

    # Set x-axis limits
    x_min = max(0, results["enrichment"].min() - results["enrichment_se"].max() * 2)
    x_max = results["enrichment"].max() + results["enrichment_se"].max() * 3
    ax.set_xlim(x_min, x_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved enrichment plot to: {output_path}")

    return fig


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Create LDSC plots")
    parser.add_argument("--type", choices=["h2", "rg", "enrichment"], required=True)
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--title", help="Plot title")

    args = parser.parse_args()

    data = pd.read_csv(args.input)

    if args.type == "h2":
        create_h2_barplot(data, output_path=args.output, title=args.title)
    elif args.type == "rg":
        create_rg_heatmap(data, output_path=args.output, title=args.title)
    elif args.type == "enrichment":
        create_enrichment_plot(data, output_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
