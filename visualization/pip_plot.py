"""
pip_plot.py - Posterior Inclusion Probability visualization

Creates bar charts and other visualizations of PIP values from SuSiE results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_pip_barplot(
    results: pd.DataFrame,
    top_n: int = 20,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Posterior Inclusion Probabilities",
    highlight_threshold: float = 0.5,
) -> plt.Figure:
    """
    Create a horizontal bar plot of top PIP variants.

    Args:
        results: DataFrame with columns SNP, PIP, and optionally CS
        top_n: Number of top variants to show
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title
        highlight_threshold: PIP threshold for highlighting

    Returns:
        matplotlib Figure object
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    if "PIP" not in results.columns or "SNP" not in results.columns:
        raise ValueError("Results must have SNP and PIP columns")

    # Sort by PIP and take top N
    results = results.sort_values("PIP", ascending=False).head(top_n)

    # Reverse for horizontal bar plot (highest at top)
    results = results.iloc[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors based on credible set membership
    colors = []
    cs_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

    for _, row in results.iterrows():
        if "CS" in results.columns and pd.notna(row.get("CS")):
            cs_idx = int(row["CS"]) - 1
            colors.append(cs_colors[cs_idx % len(cs_colors)])
        elif row["PIP"] >= highlight_threshold:
            colors.append("#2C3E50")  # Dark blue for high PIP
        else:
            colors.append("#95A5A6")  # Gray for low PIP

    # Create horizontal bar plot
    y_pos = np.arange(len(results))
    bars = ax.barh(y_pos, results["PIP"], color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for i, (bar, pip) in enumerate(zip(bars, results["PIP"])):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{pip:.3f}",
            va="center",
            fontsize=9,
        )

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["SNP"], fontsize=9)

    # Set x-axis
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Posterior Inclusion Probability (PIP)")

    # Add vertical lines
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="PIP = 0.5")
    ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="PIP = 0.9")

    # Title and legend
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Create legend for credible sets
    legend_elements = []
    if "CS" in results.columns:
        unique_cs = results["CS"].dropna().unique()
        for cs_id in sorted(unique_cs):
            cs_idx = int(cs_id) - 1
            legend_elements.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=cs_colors[cs_idx % len(cs_colors)], label=f"CS{int(cs_id)}"
                )
            )

    legend_elements.extend(
        [
            plt.Line2D([0], [0], color="orange", linestyle="--", label="PIP = 0.5"),
            plt.Line2D([0], [0], color="red", linestyle="--", label="PIP = 0.9"),
        ]
    )

    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved PIP bar plot to: {output_path}")

    return fig


def create_pip_manhattan(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str = "PIP Manhattan Plot",
) -> plt.Figure:
    """
    Create a Manhattan-style plot of PIP values by genomic position.

    Args:
        results: DataFrame with columns SNP, CHR, BP, PIP, and optionally CS
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    required = ["SNP", "BP", "PIP"]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by position
    results = results.sort_values("BP")

    fig, ax = plt.subplots(figsize=figsize)

    # Colors for credible sets
    cs_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

    # Assign colors
    colors = []
    for _, row in results.iterrows():
        if "CS" in results.columns and pd.notna(row.get("CS")):
            cs_idx = int(row["CS"]) - 1
            colors.append(cs_colors[cs_idx % len(cs_colors)])
        else:
            colors.append("#808080")

    # Plot
    bp_mb = results["BP"] / 1e6
    ax.scatter(bp_mb, results["PIP"], c=colors, s=30, alpha=0.7, edgecolors="none")

    # Highlight high PIP variants
    high_pip = results[results["PIP"] > 0.5]
    if len(high_pip) > 0:
        for _, row in high_pip.iterrows():
            ax.annotate(
                row["SNP"],
                xy=(row["BP"] / 1e6, row["PIP"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                alpha=0.8,
            )

    # Threshold lines
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Labels
    ax.set_xlabel(f"Position (Mb)")
    ax.set_ylabel("PIP")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved PIP Manhattan plot to: {output_path}")

    return fig


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Create PIP plots")
    parser.add_argument("--results", required=True, help="SuSiE results CSV")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--type", choices=["bar", "manhattan"], default="bar", help="Plot type"
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top N variants for bar plot")
    parser.add_argument("--title", help="Plot title")

    args = parser.parse_args()

    # Load data
    results = pd.read_csv(args.results)

    # Create plot
    if args.type == "bar":
        create_pip_barplot(
            results=results,
            top_n=args.top_n,
            output_path=args.output,
            title=args.title or "Posterior Inclusion Probabilities",
        )
    else:
        create_pip_manhattan(
            results=results,
            output_path=args.output,
            title=args.title or "PIP Manhattan Plot",
        )


if __name__ == "__main__":
    main()
