---
name: statgen-skills
description: >-
  Runs statistical genetics analyses including SuSiE fine-mapping, LDSC
  heritability and genetic correlations, TWAS simulation, and FUSION TWAS with
  GTEx weights. Includes JAX/Equinox coding guidelines for numerical computing.
  Use when working with GWAS summary statistics, fine-mapping, heritability
  estimation, genetic correlations, gene-trait associations, eQTL analysis, or
  writing JAX/Equinox code for statistical genetics.
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

## Visualization

- `visualization/locus_zoom.py` - Regional association plots
- `visualization/pip_plot.py` - PIP visualization
- `visualization/credible_set.py` - Credible set plots
- `visualization/ldsc_plots.py` - h² bar charts, rg heatmaps, enrichment plots
- `visualization/twas_plots.py` - Power curves, Manhattan plots, QQ plots
- `visualization/fusion_plots.py` - FUSION result plots
- `visualization/interactive_report.py` - HTML report generation

## Example Prompts

```
"Run SuSiE on my GWAS summary stats with the provided LD matrix. Use L=5 and 95% coverage."
"Estimate the SNP heritability for my height GWAS using EUR reference"
"Calculate genetic correlations between height, BMI, and educational attainment"
"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"
"Run TWAS on my schizophrenia GWAS using GTEx brain cortex"
```
