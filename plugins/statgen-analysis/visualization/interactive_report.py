"""
interactive_report.py - Generate interactive HTML reports for SuSiE results

Creates comprehensive HTML reports with interactive Plotly visualizations,
sortable tables, and interpretation guidance.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def create_interactive_pip_plot(results: pd.DataFrame) -> str:
    """Create interactive PIP scatter plot with Plotly."""
    if not HAS_PLOTLY:
        return "<p>Plotly not installed. Install with: pip install plotly</p>"

    results = results.copy()
    results.columns = results.columns.str.upper()

    # Create hover text
    hover_text = []
    for _, row in results.iterrows():
        text = f"<b>{row['SNP']}</b><br>"
        if "CHR" in row:
            text += f"Chr: {row['CHR']}<br>"
        if "BP" in row:
            text += f"Position: {int(row['BP']):,}<br>"
        text += f"PIP: {row['PIP']:.4f}<br>"
        if "P" in results.columns and pd.notna(row.get("P")):
            text += f"P-value: {row['P']:.2e}<br>"
        if "CS" in results.columns and pd.notna(row.get("CS")):
            text += f"Credible Set: {int(row['CS'])}"
        hover_text.append(text)

    # Assign colors
    if "CS" in results.columns:
        cs_colors = {1: "#E41A1C", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#FF7F00"}
        colors = [
            cs_colors.get(int(cs), "#808080") if pd.notna(cs) else "#808080"
            for cs in results["CS"]
        ]
    else:
        colors = ["#808080"] * len(results)

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=results["BP"] / 1e6 if "BP" in results.columns else results.index,
            y=results["PIP"],
            mode="markers",
            marker=dict(color=colors, size=8, opacity=0.7),
            hovertext=hover_text,
            hoverinfo="text",
            name="Variants",
        )
    )

    # Threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="PIP=0.5")
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="PIP=0.9")

    fig.update_layout(
        title="Posterior Inclusion Probabilities",
        xaxis_title="Position (Mb)" if "BP" in results.columns else "Variant Index",
        yaxis_title="PIP",
        yaxis=dict(range=[0, 1.05]),
        hovermode="closest",
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_interactive_locus_zoom(results: pd.DataFrame) -> str:
    """Create interactive locus zoom plot with Plotly."""
    if not HAS_PLOTLY:
        return "<p>Plotly not installed. Install with: pip install plotly</p>"

    results = results.copy()
    results.columns = results.columns.str.upper()

    if "P" not in results.columns:
        return "<p>P-value column not found. Cannot create locus zoom plot.</p>"

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=("-log10(P)", "PIP"),
    )

    # Calculate -log10(P)
    results["NEGLOGP"] = -np.log10(results["P"].replace(0, 1e-300))
    bp_mb = results["BP"] / 1e6

    # Colors for credible sets
    if "CS" in results.columns:
        cs_colors = {1: "#E41A1C", 2: "#377EB8", 3: "#4DAF4A", 4: "#984EA3", 5: "#FF7F00"}
        colors = [
            cs_colors.get(int(cs), "#808080") if pd.notna(cs) else "#808080"
            for cs in results["CS"]
        ]
    else:
        colors = ["#808080"] * len(results)

    # Hover text
    hover_text = [
        f"<b>{row['SNP']}</b><br>PIP: {row['PIP']:.3f}<br>P: {row['P']:.2e}"
        for _, row in results.iterrows()
    ]

    # Top panel: -log10(P)
    fig.add_trace(
        go.Scatter(
            x=bp_mb,
            y=results["NEGLOGP"],
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.7),
            hovertext=hover_text,
            hoverinfo="text",
            name="-log10(P)",
        ),
        row=1,
        col=1,
    )

    # Genome-wide significance line
    fig.add_hline(y=-np.log10(5e-8), line_dash="dash", line_color="red", row=1, col=1)

    # Bottom panel: PIP
    fig.add_trace(
        go.Bar(
            x=bp_mb,
            y=results["PIP"],
            marker=dict(color=colors, opacity=0.8),
            hovertext=hover_text,
            hoverinfo="text",
            name="PIP",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        title_text="Locus Zoom Plot",
    )

    fig.update_xaxes(title_text="Position (Mb)", row=2, col=1)
    fig.update_yaxes(title_text="-log10(P)", row=1, col=1)
    fig.update_yaxes(title_text="PIP", range=[0, 1.05], row=2, col=1)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_results_table_html(results: pd.DataFrame, max_rows: int = 100) -> str:
    """Create sortable HTML table of results."""
    results = results.copy()
    results.columns = results.columns.str.upper()

    # Select columns to display
    display_cols = ["SNP", "CHR", "BP", "PIP"]
    if "CS" in results.columns:
        display_cols.append("CS")
    if "P" in results.columns:
        display_cols.append("P")
    if "BETA" in results.columns:
        display_cols.append("BETA")
    if "SE" in results.columns:
        display_cols.append("SE")

    available_cols = [col for col in display_cols if col in results.columns]
    df = results[available_cols].head(max_rows).copy()

    # Format numeric columns
    if "PIP" in df.columns:
        df["PIP"] = df["PIP"].apply(lambda x: f"{x:.4f}")
    if "P" in df.columns:
        df["P"] = df["P"].apply(lambda x: f"{x:.2e}")
    if "BETA" in df.columns:
        df["BETA"] = df["BETA"].apply(lambda x: f"{x:.4f}")
    if "SE" in df.columns:
        df["SE"] = df["SE"].apply(lambda x: f"{x:.4f}")
    if "CS" in df.columns:
        df["CS"] = df["CS"].apply(lambda x: f"CS{int(x)}" if pd.notna(x) else "-")

    # Generate HTML table with DataTables styling
    table_html = df.to_html(
        index=False, classes="display compact", table_id="results-table", escape=False
    )

    return table_html


def generate_interpretation_section(results: pd.DataFrame, summary: dict | None = None) -> str:
    """Generate interpretation text based on results."""
    results = results.copy()
    results.columns = results.columns.str.upper()

    lines = []

    # Number of credible sets
    if "CS" in results.columns:
        n_cs = results["CS"].dropna().nunique()
        if n_cs == 0:
            lines.append(
                "<p><strong>No credible sets identified.</strong> This could indicate:</p>"
                "<ul>"
                "<li>No strong causal signal at this locus</li>"
                "<li>The signal is spread across many correlated variants</li>"
                "<li>Try increasing the L parameter or adjusting coverage</li>"
                "</ul>"
            )
        elif n_cs == 1:
            lines.append(f"<p><strong>One credible set identified.</strong> This suggests a single causal signal at this locus.</p>")
        else:
            lines.append(
                f"<p><strong>{n_cs} credible sets identified.</strong> This suggests {n_cs} independent causal signals at this locus.</p>"
            )

    # Lead variants
    high_pip = results[results["PIP"] > 0.5]
    if len(high_pip) > 0:
        lines.append("<h4>High-Confidence Variants (PIP > 0.5)</h4>")
        lines.append("<ul>")
        for _, row in high_pip.sort_values("PIP", ascending=False).head(5).iterrows():
            cs_str = f" (CS{int(row['CS'])})" if "CS" in row and pd.notna(row["CS"]) else ""
            lines.append(f"<li><strong>{row['SNP']}</strong>: PIP = {row['PIP']:.3f}{cs_str}</li>")
        lines.append("</ul>")

    # Interpretation guide
    lines.append(
        """
        <h4>Understanding the Results</h4>
        <p><strong>PIP (Posterior Inclusion Probability):</strong> Quantifies a variant's
        likelihood of being the true causal variant at this locus.</p>
        <ul>
            <li>PIP &gt; 0.9: Strong evidence of causality</li>
            <li>PIP 0.5-0.9: Moderate evidence</li>
            <li>PIP &lt; 0.5: Unlikely to be causal on its own</li>
        </ul>
        <p><strong>Credible Sets:</strong> Groups of variants that together have high
        probability of containing the causal variant. Smaller sets = better resolution.</p>
        """
    )

    return "\n".join(lines)


def generate_html_report(
    results: pd.DataFrame,
    summary: dict | None = None,
    output_path: str | Path | None = None,
    title: str = "SuSiE Fine-Mapping Results",
) -> str:
    """
    Generate a complete interactive HTML report.

    Args:
        results: DataFrame with SuSiE results
        summary: Optional summary dict from susie_summary.json
        output_path: Path to save report (optional)
        title: Report title

    Returns:
        HTML string
    """
    results = results.copy()
    results.columns = results.columns.str.upper()

    # Sort by PIP
    results = results.sort_values("PIP", ascending=False)

    # Generate sections
    pip_plot = create_interactive_pip_plot(results)
    locus_zoom = create_interactive_locus_zoom(results) if "P" in results.columns else ""
    table_html = create_results_table_html(results)
    interpretation = generate_interpretation_section(results, summary)

    # Summary stats
    n_variants = len(results)
    n_cs = results["CS"].dropna().nunique() if "CS" in results.columns else 0
    max_pip = results["PIP"].max()
    lead_variant = results.loc[results["PIP"].idxmax(), "SNP"]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
        <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h3 {{ color: #7f8c8d; }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }}
            .summary-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                color: #7f8c8d;
            }}
            .summary-card .value {{
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .plot-container {{
                margin: 20px 0;
            }}
            table.dataTable {{
                font-size: 13px;
            }}
            .interpretation {{
                background: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .footer {{
                text-align: center;
                color: #7f8c8d;
                font-size: 12px;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Variants</h3>
                    <div class="value">{n_variants:,}</div>
                </div>
                <div class="summary-card">
                    <h3>Credible Sets</h3>
                    <div class="value">{n_cs}</div>
                </div>
                <div class="summary-card">
                    <h3>Max PIP</h3>
                    <div class="value">{max_pip:.3f}</div>
                </div>
                <div class="summary-card">
                    <h3>Lead Variant</h3>
                    <div class="value" style="font-size: 16px;">{lead_variant}</div>
                </div>
            </div>
        </div>

        <div class="container">
            <h2>Interactive Visualizations</h2>

            <h3>PIP by Position</h3>
            <div class="plot-container">
                {pip_plot}
            </div>

            {"<h3>Locus Zoom</h3><div class='plot-container'>" + locus_zoom + "</div>" if locus_zoom else ""}
        </div>

        <div class="container">
            <h2>Results Table</h2>
            <p>Showing top variants sorted by PIP. Click column headers to sort.</p>
            {table_html}
        </div>

        <div class="container">
            <h2>Interpretation</h2>
            <div class="interpretation">
                {interpretation}
            </div>
        </div>

        <div class="footer">
            <p>Generated by statgen-skills | SuSiE Fine-Mapping Analysis</p>
        </div>

        <script>
            $(document).ready(function() {{
                $('#results-table').DataTable({{
                    pageLength: 25,
                    order: [[3, 'desc']]  // Sort by PIP column
                }});
            }});
        </script>
    </body>
    </html>
    """

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(html)
        print(f"Saved interactive report to: {output_path}")

    return html


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate interactive HTML report")
    parser.add_argument("--results", required=True, help="SuSiE results CSV")
    parser.add_argument("--summary", help="SuSiE summary JSON (optional)")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--title", default="SuSiE Fine-Mapping Results", help="Report title")

    args = parser.parse_args()

    # Load data
    results = pd.read_csv(args.results)

    summary = None
    if args.summary:
        with open(args.summary) as f:
            summary = json.load(f)

    # Generate report
    generate_html_report(
        results=results,
        summary=summary,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
