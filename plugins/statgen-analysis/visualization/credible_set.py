"""
credible_set.py - Credible set visualization for SuSiE results

Creates visualizations showing credible set membership, coverage, and variant details.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_credible_set_plot(
    results: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Credible Sets Summary",
) -> plt.Figure:
    """
    Create a visualization of credible sets.

    Args:
        results: DataFrame with columns SNP, PIP, CS, CS_COVERAGE
        output_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    if "CS" not in results.columns:
        raise ValueError("Results must have CS column")

    # Filter to variants in credible sets
    cs_variants = results[results["CS"].notna()].copy()

    if len(cs_variants) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No credible sets identified",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    # Get unique credible sets
    unique_cs = sorted(cs_variants["CS"].unique())
    n_cs = len(unique_cs)

    # Create figure with subplots for each CS
    fig, axes = plt.subplots(1, n_cs, figsize=(figsize[0], figsize[1]), sharey=True)

    if n_cs == 1:
        axes = [axes]

    cs_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

    for i, cs_id in enumerate(unique_cs):
        ax = axes[i]
        cs_data = cs_variants[cs_variants["CS"] == cs_id].sort_values(
            "PIP", ascending=True
        )

        color = cs_colors[int(cs_id - 1) % len(cs_colors)]

        # Horizontal bar plot
        y_pos = np.arange(len(cs_data))
        bars = ax.barh(y_pos, cs_data["PIP"], color=color, alpha=0.8, edgecolor="white")

        # Add value labels
        for bar, pip in zip(bars, cs_data["PIP"]):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{pip:.3f}",
                va="center",
                fontsize=8,
            )

        # Y-axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cs_data["SNP"], fontsize=9)

        # X-axis
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("PIP")

        # Title with coverage
        coverage = cs_data["CS_COVERAGE"].iloc[0] if "CS_COVERAGE" in cs_data.columns else 0.95
        ax.set_title(f"CS{int(cs_id)}\n({len(cs_data)} variants, {coverage:.0%} coverage)", fontsize=10)

        # Style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved credible set plot to: {output_path}")

    return fig


def create_credible_set_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of credible sets.

    Args:
        results: DataFrame with columns SNP, PIP, CS, and optionally CHR, BP

    Returns:
        DataFrame with credible set summary
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    if "CS" not in results.columns:
        return pd.DataFrame()

    cs_variants = results[results["CS"].notna()]

    if len(cs_variants) == 0:
        return pd.DataFrame()

    summary_rows = []

    for cs_id in sorted(cs_variants["CS"].unique()):
        cs_data = cs_variants[cs_variants["CS"] == cs_id]

        # Find lead variant (highest PIP)
        lead_idx = cs_data["PIP"].idxmax()
        lead_row = cs_data.loc[lead_idx]

        row = {
            "credible_set": int(cs_id),
            "n_variants": len(cs_data),
            "lead_variant": lead_row["SNP"],
            "lead_pip": lead_row["PIP"],
            "total_pip": cs_data["PIP"].sum(),
            "min_pip": cs_data["PIP"].min(),
            "max_pip": cs_data["PIP"].max(),
        }

        # Add position info if available
        if "CHR" in cs_data.columns:
            row["chr"] = cs_data["CHR"].iloc[0]
        if "BP" in cs_data.columns:
            row["start_bp"] = cs_data["BP"].min()
            row["end_bp"] = cs_data["BP"].max()
            row["span_kb"] = (row["end_bp"] - row["start_bp"]) / 1000

        # Add coverage if available
        if "CS_COVERAGE" in cs_data.columns:
            row["coverage"] = cs_data["CS_COVERAGE"].iloc[0]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_credible_set_summary(results: pd.DataFrame, detailed: bool = True) -> str:
    """
    Generate a text summary of credible sets for display.

    Args:
        results: DataFrame with SuSiE results
        detailed: Whether to include detailed variant information

    Returns:
        Formatted string summary
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    if "CS" not in results.columns:
        return "No credible set information available."

    cs_variants = results[results["CS"].notna()]

    if len(cs_variants) == 0:
        return "No credible sets were identified by SuSiE."

    lines = ["=" * 60, "CREDIBLE SETS SUMMARY", "=" * 60, ""]

    summary = create_credible_set_summary_table(results)

    for _, row in summary.iterrows():
        cs_id = int(row["credible_set"])
        lines.append(f"Credible Set {cs_id}")
        lines.append("-" * 40)
        lines.append(f"  Number of variants: {row['n_variants']}")
        lines.append(f"  Lead variant: {row['lead_variant']} (PIP = {row['lead_pip']:.3f})")
        lines.append(f"  Total PIP: {row['total_pip']:.3f}")

        if "coverage" in row:
            lines.append(f"  Coverage: {row['coverage']:.1%}")

        if "span_kb" in row:
            lines.append(f"  Genomic span: {row['span_kb']:.1f} kb")

        if detailed:
            lines.append("")
            lines.append("  Variants in this credible set:")
            cs_data = cs_variants[cs_variants["CS"] == cs_id].sort_values(
                "PIP", ascending=False
            )
            for _, var in cs_data.iterrows():
                pos_str = ""
                if "CHR" in var and "BP" in var:
                    pos_str = f" (chr{var['CHR']}:{int(var['BP'])})"
                lines.append(f"    - {var['SNP']}: PIP = {var['PIP']:.3f}{pos_str}")

        lines.append("")

    return "\n".join(lines)


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Create credible set visualizations")
    parser.add_argument("--results", required=True, help="SuSiE results CSV")
    parser.add_argument("--output", help="Output file path for plot")
    parser.add_argument("--summary", action="store_true", help="Print text summary")

    args = parser.parse_args()

    # Load data
    results = pd.read_csv(args.results)

    if args.summary:
        print(print_credible_set_summary(results))

    if args.output:
        create_credible_set_plot(results=results, output_path=args.output)


if __name__ == "__main__":
    main()
