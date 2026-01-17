"""
locus_zoom.py - Regional association plots with PIP track for SuSiE results

Creates publication-ready locus zoom plots showing:
- Top panel: -log10(P) colored by LD with lead variant
- Middle panel: PIP values from SuSiE
- Bottom panel: Gene track (optional)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def create_locus_zoom(
    results: pd.DataFrame,
    ld_matrix: np.ndarray | None = None,
    lead_variant: str | None = None,
    genes: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    highlight_cs: bool = True,
) -> plt.Figure:
    """
    Create a locus zoom plot with PIP track.

    Args:
        results: DataFrame with columns SNP, CHR, BP, P, PIP, CS
        ld_matrix: LD matrix (n_variants x n_variants), optional
        lead_variant: Lead variant SNP ID for LD coloring
        genes: DataFrame with gene annotations (optional)
                columns: gene_name, start, end, strand
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title
        highlight_cs: Whether to highlight credible set variants

    Returns:
        matplotlib Figure object
    """
    # Standardize column names
    results = results.copy()
    results.columns = results.columns.str.upper()

    # Ensure required columns
    required = ["SNP", "BP", "PIP"]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by position
    results = results.sort_values("BP").reset_index(drop=True)

    # Determine lead variant
    if lead_variant is None:
        lead_variant = results.loc[results["PIP"].idxmax(), "SNP"]
        lead_idx = results["PIP"].idxmax()
    else:
        lead_idx = results[results["SNP"] == lead_variant].index[0]

    # Calculate LD with lead variant
    if ld_matrix is not None:
        r2 = ld_matrix[lead_idx, :] ** 2
    else:
        r2 = np.full(len(results), 0.5)  # Default gray if no LD

    # Create figure with subplots
    has_pvalue = "P" in results.columns
    has_genes = genes is not None and len(genes) > 0

    n_panels = 1 + int(has_pvalue) + int(has_genes)
    height_ratios = []
    if has_pvalue:
        height_ratios.append(3)  # P-value panel
    height_ratios.append(2)  # PIP panel
    if has_genes:
        height_ratios.append(1)  # Gene panel

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, height_ratios=height_ratios, sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    bp_mb = results["BP"] / 1e6  # Convert to Mb

    # Color map for LD
    def ld_color(r2_val):
        if r2_val >= 0.8:
            return "#FF0000"  # Red
        elif r2_val >= 0.6:
            return "#FFA500"  # Orange
        elif r2_val >= 0.4:
            return "#00FF00"  # Green
        elif r2_val >= 0.2:
            return "#87CEEB"  # Light blue
        else:
            return "#0000FF"  # Blue

    colors = [ld_color(r) for r in r2]

    # Panel 1: P-value (if available)
    if has_pvalue:
        ax = axes[ax_idx]
        ax_idx += 1

        # Calculate -log10(P)
        neglog_p = -np.log10(results["P"].replace(0, 1e-300))

        # Scatter plot
        ax.scatter(bp_mb, neglog_p, c=colors, s=30, alpha=0.8, edgecolors="none")

        # Highlight lead variant
        lead_row = results[results["SNP"] == lead_variant].iloc[0]
        lead_neglog_p = -np.log10(lead_row["P"])
        ax.scatter(
            [lead_row["BP"] / 1e6],
            [lead_neglog_p],
            c="purple",
            s=100,
            marker="D",
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )

        # Annotate lead variant
        ax.annotate(
            lead_variant,
            xy=(lead_row["BP"] / 1e6, lead_neglog_p),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

        # Genome-wide significance line
        ax.axhline(y=-np.log10(5e-8), color="red", linestyle="--", alpha=0.5)

        ax.set_ylabel(r"$-\log_{10}(P)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 2: PIP
    ax = axes[ax_idx]
    ax_idx += 1

    # Bar plot for PIP
    bar_colors = []
    for i, row in results.iterrows():
        if highlight_cs and pd.notna(row.get("CS")):
            cs_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]
            cs_idx = int(row["CS"]) - 1
            bar_colors.append(cs_colors[cs_idx % len(cs_colors)])
        else:
            bar_colors.append(colors[i])

    ax.bar(bp_mb, results["PIP"], width=0.001, color=bar_colors, alpha=0.8)

    # Highlight credible set variants with markers
    if highlight_cs and "CS" in results.columns:
        cs_variants = results[results["CS"].notna()]
        for cs_id in cs_variants["CS"].unique():
            cs_data = cs_variants[cs_variants["CS"] == cs_id]
            ax.scatter(
                cs_data["BP"] / 1e6,
                cs_data["PIP"],
                s=50,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label=f"CS{int(cs_id)}",
            )

        ax.legend(loc="upper right", fontsize=8)

    ax.set_ylabel("PIP")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: Gene track (if available)
    if has_genes:
        ax = axes[ax_idx]

        y_pos = 0.5
        gene_height = 0.3

        for _, gene in genes.iterrows():
            gene_start = gene["start"] / 1e6
            gene_end = gene["end"] / 1e6

            # Gene body
            rect = Rectangle(
                (gene_start, y_pos - gene_height / 2),
                gene_end - gene_start,
                gene_height,
                facecolor="#4169E1",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Gene name
            gene_mid = (gene_start + gene_end) / 2
            ax.text(
                gene_mid,
                y_pos,
                gene["gene_name"],
                ha="center",
                va="center",
                fontsize=7,
                style="italic",
            )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Genes")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    # X-axis label
    axes[-1].set_xlabel(f"Position on chromosome {results['CHR'].iloc[0]} (Mb)")

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    # LD legend
    ld_legend_elements = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#FF0000", markersize=8, label="0.8-1.0"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#FFA500", markersize=8, label="0.6-0.8"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#00FF00", markersize=8, label="0.4-0.6"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#87CEEB", markersize=8, label="0.2-0.4"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#0000FF", markersize=8, label="0.0-0.2"
        ),
    ]

    if has_pvalue:
        axes[0].legend(
            handles=ld_legend_elements,
            title=r"$r^2$",
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
        )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved locus zoom plot to: {output_path}")

    return fig


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Create locus zoom plot")
    parser.add_argument("--results", required=True, help="SuSiE results CSV")
    parser.add_argument("--ld", help="LD matrix file (.npy)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--title", help="Plot title")

    args = parser.parse_args()

    # Load data
    results = pd.read_csv(args.results)

    ld_matrix = None
    if args.ld:
        ld_matrix = np.load(args.ld)

    # Create plot
    create_locus_zoom(
        results=results,
        ld_matrix=ld_matrix,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
