# LDSC Tutorial: Heritability and Genetic Correlation

This tutorial demonstrates how to use LDSC within the statgen_skills toolkit.

## Prerequisites

Make sure you have the LDSC package installed:
```bash
pip install git+https://github.com/CBIIT/ldsc.git
```

## Step 1: Munge Summary Statistics

LDSC requires summary statistics in a specific format. Use `munge_sumstats` to convert:

```python
from scripts.ldsc.munge import munge_sumstats

result = munge_sumstats(
    input_path="examples/ldsc_example_sumstats.csv",
    output_prefix="output/example_munged",
    n=50000,  # Sample size if not in file
)

print(f"Munged stats saved to: {result['output_path']}")
print(f"SNPs: {result['n_snps_input']} -> {result['n_snps_output']}")
```

## Step 2: Estimate Heritability

```python
from scripts.ldsc.run_ldsc import estimate_heritability

result = estimate_heritability(
    sumstats="output/example_munged.sumstats.gz",
    output_dir="output/h2",
    population="EUR",  # Uses 1000G EUR reference
)

if result["success"]:
    h2 = result["results"]["h2"]
    se = result["results"]["h2_se"]
    print(f"SNP heritability: {h2:.4f} (SE: {se:.4f})")
else:
    print(f"Error: {result['error']}")
```

## Step 3: Genetic Correlation

To calculate genetic correlation between two traits:

```python
from scripts.ldsc.run_ldsc import genetic_correlation

result = genetic_correlation(
    sumstats=[
        "output/trait1.sumstats.gz",
        "output/trait2.sumstats.gz",
    ],
    output_dir="output/rg",
    population="EUR",
)

if result["success"]:
    for corr in result["results"]["correlations"]:
        print(f"rg = {corr['rg']:.3f} (SE: {corr['se']:.3f}), p = {corr['p']:.2e}")
```

## Step 4: Visualize Results

```python
from visualization.ldsc_plots import create_h2_barplot, create_rg_heatmap
import pandas as pd

# Heritability bar plot
h2_results = pd.DataFrame({
    "trait": ["Height", "BMI", "T2D"],
    "h2": [0.45, 0.25, 0.15],
    "h2_se": [0.05, 0.03, 0.02],
})
create_h2_barplot(h2_results, output_path="output/h2_plot.png")

# Genetic correlation heatmap
rg_matrix = pd.DataFrame(
    [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]],
    index=["Height", "BMI", "T2D"],
    columns=["Height", "BMI", "T2D"],
)
create_rg_heatmap(rg_matrix, output_path="output/rg_heatmap.png")
```

## Interpretation Guide

### Heritability (h²)
- **h² = 0.5**: 50% of trait variance explained by common SNPs
- **Intercept ≈ 1**: Good quality control, no confounding
- **Intercept > 1.1**: Possible population stratification

### Genetic Correlation (rg)
- **rg = 1**: Perfect positive genetic correlation
- **rg = 0**: No shared genetic basis
- **rg = -1**: Perfect negative genetic correlation

### Enrichment (s-LDSC)
- **Enrichment > 1**: Category has more heritability than expected
- **Enrichment = 1**: Category has expected heritability
- **Enrichment < 1**: Category has less heritability than expected

## Common Issues

### "Reference data not found"
Run `download_reference("EUR")` to download 1000G reference LD scores:

```python
from scripts.ldsc.reference_data import download_reference
download_reference("EUR", verbose=True)
```

### "Sample size (N) not found"
Provide the sample size explicitly:

```python
result = munge_sumstats(
    input_path="your_sumstats.csv",
    output_prefix="output/munged",
    n=50000,  # Add your sample size here
)
```

### "Missing required columns"
Ensure your summary statistics have: SNP, A1, A2, and either BETA/SE or Z.

## API Reference

### munge_sumstats()
```python
munge_sumstats(
    input_path,        # Path to input file
    output_prefix,     # Output file prefix
    n=None,            # Sample size (if not in file)
    info_min=0.9,      # Minimum INFO score
    maf_min=0.01,      # Minimum MAF
)
```

### estimate_heritability()
```python
estimate_heritability(
    sumstats,              # Path to munged sumstats
    output_dir,            # Output directory
    population="EUR",      # Reference population
    sample_prevalence=None,  # For case-control
    population_prevalence=None,
    no_intercept=False,    # Constrain intercept to 1
)
```

### genetic_correlation()
```python
genetic_correlation(
    sumstats,          # List of paths (at least 2)
    output_dir,        # Output directory
    population="EUR",  # Reference population
)
```

### partitioned_heritability()
```python
partitioned_heritability(
    sumstats,          # Path to munged sumstats
    output_dir,        # Output directory
    annotations,       # Path to annotation LD scores
    population="EUR",  # Reference population
)
```
