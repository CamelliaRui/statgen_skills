---
name: statgen-skills
description: Statistical genetics toolkit for fine-mapping (SuSiE), LD score regression (LDSC), and TWAS simulation, with support for GWAS summary statistics, heritability estimation, genetic correlations, and publication-ready reports
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

### TWAS Simulator

Simulate Transcriptome-Wide Association Studies for methods development, power analysis, and teaching.

**Capabilities:**
- Simulate gene expression with configurable cis-heritability
- Multiple expression prediction models (Elastic Net, LASSO, GBLUP, oracle)
- Full TWAS pipeline: expression → weights → association
- Power and FDR calculation
- Publication-ready visualizations (power curves, Manhattan, QQ plots)

**API Functions:**
- `simulate_twas(genotypes, n_causal_genes, ...)` - Run complete simulation
- `simulate_expression(genotypes, h2_cis, n_causal)` - Simulate expression only
- `run_twas(pred_expression, phenotype)` - TWAS association test
- `get_model(name)` - Get expression prediction model

**Example Usage:**

```
# Basic simulation
"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"

# Power analysis
"Run TWAS power analysis varying eQTL sample size from 100 to 1000"

# Model comparison
"Compare Elastic Net vs LASSO for TWAS prediction"
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h2_cis` | 0.1 | Cis-heritability of expression |
| `h2_trait` | 0.5 | Total trait heritability |
| `prop_mediated` | 0.1 | Fraction of h² mediated through expression |
| `n_causal_cis` | 1 | Causal cis-eQTLs per gene |
| `n_causal_genes` | 10 | Genes with trait effects |

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
- **Power curve**: TWAS power vs parameter values
- **TWAS Manhattan**: Gene-level association plot
- **QQ plot**: P-value calibration with genomic inflation

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

### TWAS
- `scripts/twas/simulate.py` - Main TWAS simulation orchestrator
- `scripts/twas/expression.py` - Expression simulation
- `scripts/twas/association.py` - TWAS association testing
- `scripts/twas/genotype.py` - Genotype loading and processing
- `scripts/twas/models/` - Expression prediction models (ElasticNet, LASSO, GBLUP)

## Visualization

- `visualization/locus_zoom.py` - Regional association plots
- `visualization/pip_plot.py` - PIP visualization
- `visualization/credible_set.py` - Credible set plots
- `visualization/interactive_report.py` - HTML report generation
- `visualization/ldsc_plots.py` - h² bar charts, rg heatmaps, enrichment plots
- `visualization/twas_plots.py` - Power curves, Manhattan plots, QQ plots

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

### FUSION TWAS

Run Transcriptome-Wide Association Studies using FUSION to identify genes whose expression is associated with complex traits.

**Capabilities:**
- Association testing with pre-computed GTEx v8 weights (49 tissues)
- Custom expression weight computation from eQTL data
- Conditional analysis to identify independent signals
- Multi-chromosome parallel execution
- Automatic reference data download and caching

**API Functions:**
- `run_twas_association(sumstats, tissue, output_dir)` - Run TWAS
- `list_available_tissues()` - List GTEx v8 tissues
- `download_weights(tissue)` - Download expression weights
- `check_dependencies()` - Check R, FUSION, PLINK availability

**Example Usage:**

```
# Basic TWAS
"Run TWAS on my height GWAS using GTEx whole blood"

# Multi-tissue analysis
"Run TWAS across brain tissues for my schizophrenia GWAS"

# Check significant genes
"Which genes are significant in my TWAS results?"
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tissue` | required | GTEx tissue name |
| `population` | EUR | LD reference population |
| `chromosomes` | 1-22 | Chromosomes to analyze |
| `gwas_n` | None | Sample size (if not in file) |
| `coloc` | False | Run colocalization |

**Requirements:**
- R 4.0+ with packages: optparse, glmnet
- PLINK (for custom weights only)

### Presentation Generator

Generate scientific presentations from research papers (PDF) with customizable templates and formats.

**Capabilities:**
- Extract content and figures from paper PDFs
- Apply custom template styling (logo, colors, fonts)
- Support for Journal Club, Lab Meeting, and Conference Talk formats
- IMRAD or flexible section structures
- Extractive or generative bullet point styles
- Export to PPTX (compatible with Google Slides)

**API Functions:**
- `generate_presentation(pdf, output, type, ...)` - Generate presentation
- `PresentationGenerator(config)` - Full control over generation
- `load_config(name)` - Load preset configuration

**Example Usage:**

```python
from scripts.presentation import generate_presentation

# Generate a journal club presentation
generate_presentation(
    pdf_path="paper.pdf",
    output_path="presentation.pptx",
    presentation_type="journal_club",
    presenter_name="Your Name",
)
```

**Interactive Workflow:**
```
"Create a journal club presentation from this paper"
"Generate a 15-minute conference talk from my manuscript"
"Make a lab meeting presentation with all supplementary figures"
```

**Presentation Types:**

| Type | Duration | Detail Level | Default Slides |
|------|----------|--------------|----------------|
| Journal Club | 30-45 min | Standard | ~20 |
| Lab Meeting | ~60 min | High (methods, implementation) | ~30 |
| Conference Talk | 15-30 min | High-level | ~12-25 |

## Future Tools (Planned)

- **TWAS** - Transcriptome-wide association studies (real data analysis)
