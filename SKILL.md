---
name: statgen-skills
description: >-
  Use when analyzing GWAS for causal variants, estimating genetic architecture,
  mapping gene-trait associations, or writing statistical genetics code in
  JAX/Equinox. Capabilities include fine-mapping (SuSiE), heritability (LDSC),
  TWAS with GTEx weights, and tissue-specific eQTL analysis.
---

# Statistical Genetics Skills

## Tools

### SuSiE Fine-Mapping

Identify causal variants at GWAS loci using Sum of Single Effects regression. Supports summary statistics or individual-level data, outputs credible sets and per-variant PIPs.

See [reference/susie.md](reference/susie.md) for parameters, scripts, and workflow.

### LDSC (LD Score Regression)

Estimate SNP heritability, genetic correlations between traits, and partition heritability by functional annotations. Supports EUR, EAS, AFR, SAS, AMR populations.

**API:** `estimate_heritability()`, `genetic_correlation()`, `partitioned_heritability()`, `munge_sumstats()`

See [reference/ldsc.md](reference/ldsc.md) for parameters, scripts, and workflow.

### TWAS Simulator

Simulate transcriptome-wide association studies for methods development and power analysis. Pluggable models: Elastic Net, LASSO, GBLUP, oracle.

**API:** `simulate_twas()`, `simulate_expression()`, `run_twas()`, `get_model()`

See [reference/twas-sim.md](reference/twas-sim.md) for parameters, models, and workflow.

### FUSION TWAS

Run TWAS with pre-computed GTEx v8 expression weights (49 tissues) to find genes associated with complex traits.

**API:** `run_twas_association()`, `list_available_tissues()`, `download_weights()`, `check_dependencies()`

See [reference/fusion.md](reference/fusion.md) for parameters, requirements, and workflow.

## JAX/Equinox Development

Guidelines for writing numerical code with JAX and Equinox: module patterns (abstract/final), JIT boundaries, PRNG discipline, PyTree stability, numerics, and linear algebra. Includes checklists and code snippets.

See [reference/jax-equinox.md](reference/jax-equinox.md) for rules, checklists, and ready-to-use patterns.

## Input Formats

See [reference/input-formats.md](reference/input-formats.md) for summary statistics columns, LD matrix options, individual-level data formats, and output descriptions.

## Scientific Figure Generation

### Data Integrity

All figures must use real data. Never silently substitute simulated or synthetic data. If real data is unavailable, stop and warn the user before proceeding. Label any approved simulation output with "SIMULATED DATA" in the plot title and filename.

### Built-in Plot Functions

Import from `visualization/`:

```python
from visualization import (
    # SuSiE fine-mapping
    create_locus_zoom,          # Regional association plot with LD coloring + PIP track
    create_pip_barplot,         # PIP bar chart colored by credible set
    create_pip_manhattan,       # Manhattan-style PIP plot by genomic position
    create_credible_set_plot,   # Multi-panel credible set visualization
    # LDSC
    create_h2_barplot,          # Heritability estimates with 95% CI error bars
    create_rg_heatmap,          # Genetic correlation heatmap (lower triangle)
    create_enrichment_plot,     # s-LDSC enrichment forest plot
    # TWAS
    create_power_curve,         # Power vs parameter curves with error bands
    create_model_comparison,    # Model performance bar chart
    create_twas_manhattan,      # TWAS Manhattan plot
    create_qq_plot,             # QQ plot with genomic inflation factor
    # FUSION
    create_fusion_locus_plot,   # GWAS + TWAS dual-panel locus plot
    create_tissue_heatmap,      # Gene-by-tissue Z-score heatmap
    # Reports
    generate_html_report,       # Interactive HTML report (Plotly)
)
```

All functions accept a `results` DataFrame, optional `output_path`, `figsize`, and `title`. They return a `matplotlib.Figure`.

### Style Conventions

Follow these defaults when creating new figures or extending existing ones:

| Setting | Value |
|---|---|
| DPI | 300 (publication standard) |
| Format | PDF or PNG (never JPEG for scientific plots) |
| Font | Arial / Helvetica, sans-serif |
| Axis labels | 8–9 pt, sentence case, with units in parentheses |
| Tick labels | 7–8 pt |
| Titles | 10–12 pt, `fontweight="bold"` |
| Spines | Remove top and right (`ax.spines['top'].set_visible(False)`) |
| Layout | `plt.tight_layout()` or `bbox_inches='tight'` on save |
| Error bars | 95% CI (SE × 1.96), specify type in caption |

### Color Palettes

**Categorical data** — use colorblind-safe Okabe-Ito:
```python
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
```

**Credible sets** — 5-color palette already used in `pip_plot.py` and `credible_set.py`:
```python
cs_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
```

**LD r²** — red (≥0.8) → orange → green → light blue → blue (<0.2).

**Heatmaps / continuous data** — perceptually uniform (`viridis`, `cividis`); for diverging data use `RdBu_r` centered at 0. Never use `jet` or `rainbow`.

### Figure Sizing for Journals

| Journal | Single column | Double column |
|---|---|---|
| Nature | 89 mm (3.5 in) | 183 mm (7.2 in) |
| Science | 55 mm (2.2 in) | 175 mm (6.9 in) |
| Cell | 85 mm (3.3 in) | 178 mm (7.0 in) |

### Multi-Panel Figures

Use `GridSpec` for flexible layouts. Label panels with bold uppercase letters (A, B, C):
```python
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top')
```

### Saving Figures (Python)

```python
fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')
fig.savefig('figure1.png', dpi=300, bbox_inches='tight')
```

### R / ggplot2 Figures

#### Theme Defaults

Use `theme_classic` as the base and override consistently:

```r
theme_classic(base_size = 14) +
  theme(
    plot.title    = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title    = element_text(size = 14),
    axis.text     = element_text(size = 12),
    legend.title  = element_text(size = 12),
    legend.text   = element_text(size = 11),
    panel.grid.major.y = element_line(color = "gray90", linetype = "dotted"),
    panel.grid.major.x = element_blank(),
    plot.margin   = margin(t = 20, r = 20, b = 20, l = 40)
  )
```

- Legend titles should be lowercase (e.g., `"method"` not `"Method"`).
- Legend position: bottom or right as appropriate.

#### Panel Labels

Use a dedicated helper function for panel labels (A, B, C). Do not use `annotate("text")`, `Inf` coordinates, or hardcoded numeric positions. Panel labels belong at the **figure level**, not inside the ggplot coordinate system.

```r
# Define or source a panel label function
add_panel_label <- function(p, label, x = 0.02, y = 0.99) {
  p + annotation_custom(
    grid::textGrob(label, x = x, y = y,
                   gp = grid::gpar(fontsize = 16, fontface = "bold")),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  )
}

p <- add_panel_label(p, "A")
```

#### Axis Formatting

- Numeric x-axes: always set explicit breaks with `scale_x_continuous(breaks = ...)`.
- Y-axis padding: `expand = expansion(mult = c(0, 0.1))` to avoid clipping.
- Do not alter axis limits unless explicitly needed.

#### Output Format

Save R figures as PNG only:

```r
ggsave("figure1.png", plot = p, width = 8, height = 6, dpi = 300)
```

#### Modification Policy

When editing an existing R figure script:

1. Read the existing code first.
2. Apply **minimal** changes — do not regenerate the full plot.
3. Do not rewrite data loading, theme, axis code, or geoms unless asked.
4. Keep exactly the subplots and panels requested — no extra panels or unsolicited summary statistics.

### Checklist Before Submission

- Resolution ≥ 300 DPI, vector format preferred
- All axes labeled with units
- Colorblind-friendly palette, interpretable in grayscale
- Error bars present with type stated in caption
- Text readable at final print size (≥ 6 pt)
- No 3D effects, chart junk, or truncated y-axes on bar charts
- Data source is real (not simulated) unless explicitly approved

## Example Prompts

```
"Run SuSiE on my GWAS summary stats with the provided LD matrix. Use L=5 and 95% coverage."
"Estimate the SNP heritability for my height GWAS using EUR reference"
"Calculate genetic correlations between height, BMI, and educational attainment"
"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"
"Run TWAS on my schizophrenia GWAS using GTEx brain cortex"
```
