---
name: statgen-skills
description: Statistical genetics toolkit for fine-mapping (SuSiE) and LD score regression (LDSC), with support for GWAS summary statistics, heritability estimation, genetic correlations, and publication-ready reports
---

# Statistical Genetics Skills

A comprehensive toolkit for genetic fine-mapping and downstream analysis, designed for both experienced geneticists and newcomers to the field.

## Current Tools

### SuSiE Fine-Mapping

Run Sum of Single Effects (SuSiE) regression to identify causal variants at GWAS loci.

**Capabilities:**
- Fine-mapping with GWAS summary statistics (Z-scores or beta/SE)
- Fine-mapping with individual-level genotype + phenotype data
- Automatic credible set construction with configurable coverage
- Detection of multiple independent causal signals
- Per-variant posterior inclusion probabilities (PIP)

### LDSC (LD Score Regression)

Estimate SNP heritability, genetic correlations, and partition heritability by functional annotations.

**Capabilities:**
- SNP heritability (h²) estimation from GWAS summary statistics
- Genetic correlation (rg) between multiple traits
- Stratified LDSC (s-LDSC) for heritability partitioning by annotations
- Automatic reference data download for EUR, EAS, AFR, SAS, AMR populations
- Publication-ready visualization (h² bar charts, rg heatmaps, enrichment plots)

**API Functions:**
- `estimate_heritability(sumstats, output_dir, population)` - SNP h² from GWAS
- `genetic_correlation(sumstats_list, output_dir, population)` - rg between traits
- `partitioned_heritability(sumstats, annotations, output_dir)` - s-LDSC enrichment
- `munge_sumstats(input, output)` - Convert summary stats to LDSC format

**Example Usage:**

```
# Estimate heritability
"Estimate the SNP heritability for my height GWAS using EUR reference"

# Genetic correlation
"Calculate genetic correlations between height, BMI, and educational attainment"

# Partitioned heritability
"Partition heritability for schizophrenia using baseline-LD annotations"
```

**Key Concepts:**

- **SNP heritability (h²):** Proportion of trait variance explained by common SNPs
- **Genetic correlation (rg):** Shared genetic architecture between traits (-1 to 1)
- **Enrichment:** How much more heritability is in an annotation than expected by SNP count
- **Intercept:** Quality control metric; values > 1 suggest population stratification

## Input Formats

### Summary Statistics

Provide a CSV/TSV file with columns:
- `SNP` - Variant identifier (rsID)
- `CHR` - Chromosome
- `BP` - Base pair position
- `A1` - Effect allele
- `A2` - Other allele
- `BETA` and `SE` - Effect size and standard error, OR
- `Z` - Z-score
- `P` - P-value (optional, for visualization)

### LD Matrix

Three options:
1. **User-provided**: Supply a pre-computed LD matrix (numpy .npy or text)
2. **Reference panel**: Specify population (e.g., "1000G_EUR", "1000G_EAS")
3. **Compute from data**: Automatically computed when individual-level data provided

### Individual-Level Data

- PLINK format: `.bed/.bim/.fam` files
- VCF format: `.vcf.gz` with separate phenotype file
- Phenotype file: Tab-delimited with FID, IID, PHENO columns

## Analysis Parameters

### SuSiE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 10 | Maximum number of causal variants |
| `coverage` | 0.95 | Credible set coverage probability |
| `min_abs_corr` | 0.5 | Minimum LD for credible set membership |

### LDSC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population` | EUR | Reference population (EUR, EAS, AFR, SAS, AMR) |
| `no_intercept` | False | Constrain intercept to 1 |
| `samp_prev` | None | Sample prevalence (case-control) |
| `pop_prev` | None | Population prevalence (case-control) |

## Outputs

### Tables
- **CSV**: Results with SNP, PIP, credible set membership
- **Excel**: Multi-sheet workbook with variants, credible sets, and summary

### Figures
- **Locus zoom**: Regional association plot with PIP track and gene annotations
- **PIP plot**: Bar chart of posterior inclusion probabilities
- **Credible set visualization**: Variants grouped by credible set
- **h² bar plot**: Heritability estimates with confidence intervals
- **rg heatmap**: Genetic correlation matrix with significance
- **Enrichment plot**: Forest plot of s-LDSC enrichments

### Reports
- **Interactive HTML**: All figures with hover details, sortable tables, interpretation guide

## Example Usage

### Quick Analysis (Experts)

```
"Run SuSiE on my GWAS summary stats with the provided LD matrix. Use L=5 and 95% coverage."
```

### Guided Analysis (Newcomers)

```
"I have GWAS results for a locus on chromosome 6. Help me identify the causal variant using fine-mapping. Please explain what the results mean."
```

### Specific Tasks

```
"Fine-map the MHC region using EUR 1000G reference LD"

"Generate a publication-ready locus zoom plot for my SuSiE results"

"What's the probability that rs12345 is the causal variant at this locus?"
```

## Key Concepts

### Posterior Inclusion Probability (PIP)

PIP quantifies a variant's likelihood of being the true causal variant at a genetic locus.

- **PIP > 0.9**: Strong evidence this variant is causal
- **PIP 0.5-0.9**: Moderate evidence
- **PIP < 0.5**: Unlikely to be causal on its own

In regions with high LD, the causal signal may spread across correlated variants, resulting in moderate PIPs for several variants.

### Credible Sets

A 95% credible set is a group of variants that together have a 95% probability of containing the true causal variant. Smaller credible sets indicate better resolution.

## Scripts

### SuSiE
- `scripts/susie/run_susie.R` - Core SuSiE execution wrapper
- `scripts/ld/compute_ld.R` - LD matrix computation
- `scripts/ld/fetch_ld_ref.py` - Reference panel retrieval
- `scripts/utils/validate_input.py` - Input validation

### LDSC
- `scripts/ldsc/run_ldsc.py` - Main entry point for all LDSC analyses
- `scripts/ldsc/munge.py` - Summary statistics preprocessing
- `scripts/ldsc/reference_data.py` - Reference data management
- `scripts/ldsc/parsers.py` - Log file parsing

## Visualization

- `visualization/locus_zoom.py` - Regional association plots
- `visualization/pip_plot.py` - PIP visualization
- `visualization/credible_set.py` - Credible set plots
- `visualization/interactive_report.py` - HTML report generation
- `visualization/ldsc_plots.py` - h² bar charts, rg heatmaps, enrichment plots

## Best Practices

1. **Validate your LD matrix** - Mismatched LD is a common source of errors
2. **Start with default L** - Increase only if you expect many independent signals
3. **Check convergence** - SuSiE reports if the algorithm converged
4. **Consider LD structure** - High LD regions may have larger credible sets
5. **Use matching populations** - LD reference should match your GWAS population

## Limitations

- Requires accurate LD information matching the GWAS population
- Assumes at most L causal variants (may miss signals if L is too low)
- Summary statistics analysis assumes linear model
- Not suitable for highly polygenic traits with many small effects

## Future Tools (Planned)

- **TWAS** - Transcriptome-wide association studies
- **TWAS simulator** - Simulate TWAS data for methods development
