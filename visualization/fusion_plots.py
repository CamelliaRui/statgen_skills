"""
fusion_plots.py - FUSION TWAS visualization functions

Creates visualization plots for FUSION TWAS results including:
- Regional locus plots combining GWAS and TWAS data
- Multi-tissue heatmaps of TWAS Z-scores
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_fusion_locus_plot(
    twas_df: pd.DataFrame,
    gwas_df: pd.DataFrame,
    chromosome: int,
    region_start: int,
    region_end: int,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    significance_threshold: float = 5e-8,
) -> plt.Figure:
    """
    Create a combined GWAS+TWAS regional locus plot.

    Creates a two-panel figure with:
    - Top panel: GWAS -log10(P) scatter plot
    - Bottom panel: TWAS genes as horizontal bars colored by significance/direction

    Args:
        twas_df: DataFrame with columns gene, chromosome, start, end, twas_z, twas_p
        gwas_df: DataFrame with columns SNP, BP, P
        chromosome: Chromosome number for the region
        region_start: Start position of the region
        region_end: End position of the region
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title (optional)
        significance_threshold: GWAS significance threshold for horizontal line

    Returns:
        matplotlib Figure object
    """
    # Filter GWAS data to region
    gwas_filtered = gwas_df[
        (gwas_df["BP"] >= region_start) & (gwas_df["BP"] <= region_end)
    ].copy()

    # Filter TWAS data to chromosome and region
    twas_filtered = twas_df[
        (twas_df["chromosome"] == chromosome)
        & (twas_df["start"] <= region_end)
        & (twas_df["end"] >= region_start)
    ].copy()

    # Create figure with two subplots
    fig, (ax_gwas, ax_twas) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True
    )

    # --- Top panel: GWAS scatter plot ---
    if len(gwas_filtered) > 0:
        gwas_filtered["neglog10p"] = -np.log10(gwas_filtered["P"])
        ax_gwas.scatter(
            gwas_filtered["BP"] / 1e6,
            gwas_filtered["neglog10p"],
            c="#1f77b4",
            s=20,
            alpha=0.6,
            edgecolors="none",
        )

        # Add significance threshold line
        sig_line = -np.log10(significance_threshold)
        ax_gwas.axhline(
            y=sig_line, color="red", linestyle="--", alpha=0.7, linewidth=1
        )
        ax_gwas.text(
            region_end / 1e6,
            sig_line,
            f"  P={significance_threshold:.0e}",
            va="center",
            fontsize=8,
            color="red",
        )

    ax_gwas.set_ylabel("-log10(P)")
    ax_gwas.set_ylim(bottom=0)
    ax_gwas.spines["top"].set_visible(False)
    ax_gwas.spines["right"].set_visible(False)

    # --- Bottom panel: TWAS gene bars ---
    if len(twas_filtered) > 0:
        # Sort genes by start position
        twas_filtered = twas_filtered.sort_values("start")

        # Assign y positions for genes
        y_positions = np.arange(len(twas_filtered))

        # Determine colors based on Z-score direction and significance
        colors = []
        for _, row in twas_filtered.iterrows():
            z = row["twas_z"]
            p = row.get("twas_p", 1)

            # Significant genes (p < 0.05) get saturated colors
            if p < 0.05:
                if z > 0:
                    colors.append("#d62728")  # Red for positive Z
                else:
                    colors.append("#2ca02c")  # Green for negative Z
            else:
                colors.append("#808080")  # Gray for non-significant

        # Draw horizontal bars for genes
        for i, (_, row) in enumerate(twas_filtered.iterrows()):
            ax_twas.barh(
                y=i,
                width=(row["end"] - row["start"]) / 1e6,
                left=row["start"] / 1e6,
                height=0.6,
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
            )

            # Add gene labels
            gene_center = (row["start"] + row["end"]) / 2 / 1e6
            ax_twas.text(
                gene_center,
                i,
                row["gene"],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

        ax_twas.set_yticks(y_positions)
        ax_twas.set_yticklabels(
            [f"Z={z:.1f}" for z in twas_filtered["twas_z"]], fontsize=8
        )
        ax_twas.set_ylim(-0.5, len(twas_filtered) - 0.5)

    ax_twas.set_xlabel(f"Chromosome {chromosome} Position (Mb)")
    ax_twas.set_xlim(region_start / 1e6, region_end / 1e6)
    ax_twas.spines["top"].set_visible(False)
    ax_twas.spines["right"].set_visible(False)

    # Set title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    else:
        fig.suptitle(
            f"FUSION Locus Plot - Chr{chromosome}:{region_start:,}-{region_end:,}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#d62728", label="TWAS Z > 0 (sig)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", label="TWAS Z < 0 (sig)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#808080", label="Non-significant"),
    ]
    ax_twas.legend(
        handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9
    )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved FUSION locus plot to: {output_path}")

    return fig


def create_tissue_heatmap(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str = "TWAS Z-scores Across Tissues",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Create a heatmap of TWAS Z-scores across multiple tissues.

    Args:
        results: DataFrame with columns gene, tissue, twas_z
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for color scale (auto if None)
        vmax: Maximum value for color scale (auto if None)

    Returns:
        matplotlib Figure object
    """
    # Pivot data to create gene x tissue matrix
    heatmap_data = results.pivot(index="gene", columns="tissue", values="twas_z")

    # Determine color scale limits if not provided
    if vmin is None or vmax is None:
        max_abs = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "TWAS Z-score"},
    )

    # Set labels and title
    ax.set_xlabel("Tissue", fontsize=10)
    ax.set_ylabel("Gene", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved tissue heatmap to: {output_path}")

    return fig
