# SuSiE Fine-Mapping Workflow Example

This guide walks through a complete fine-mapping analysis using the statgen-skills toolkit.

## Prerequisites

### R Dependencies

```r
install.packages(c("susieR", "data.table", "jsonlite", "optparse"))
```

### Python Dependencies

```bash
pip install pandas numpy matplotlib plotly openpyxl
```

## Example 1: Basic Fine-Mapping with Summary Statistics

### Step 1: Prepare Your Data

Your summary statistics file should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| SNP | Variant ID (rsID) | Yes |
| CHR | Chromosome | Yes |
| BP | Position | Yes |
| A1 | Effect allele | Recommended |
| A2 | Other allele | Recommended |
| BETA | Effect size | Yes (or Z) |
| SE | Standard error | Yes (if BETA) |
| Z | Z-score | Yes (or BETA+SE) |
| P | P-value | Recommended |

Example file: `example_sumstats.csv`

### Step 2: Prepare LD Matrix

The LD matrix should be:
- Square matrix (n_variants x n_variants)
- Same variant order as summary statistics
- Correlation (r), not r²

Options:
1. **Pre-computed**: Provide as `.npy`, `.txt`, or `.rds` file
2. **From reference**: Specify `1000G_EUR`, `1000G_EAS`, etc.

### Step 3: Run SuSiE

Using the R wrapper script:

```bash
Rscript scripts/susie/run_susie.R \
    --input examples/example_sumstats.csv \
    --input-type sumstats \
    --ld path/to/ld_matrix.npy \
    --n 50000 \
    --output results/ \
    --L 10 \
    --coverage 0.95 \
    --verbose
```

Or through Claude with the skill:

```
"Run SuSiE fine-mapping on my GWAS summary statistics in example_sumstats.csv.
Sample size is 50,000. Use the provided LD matrix. Generate all outputs."
```

### Step 4: Interpret Results

Output files:
- `susie_results.csv` - Per-variant results with PIP and CS membership
- `susie_summary.json` - Summary statistics and credible set info
- `susie_fit.rds` - Full R object for advanced analysis

Key columns in results:
- **PIP**: Posterior probability this variant is causal
- **CS**: Credible set number (1, 2, ...) or NA
- **CS_COVERAGE**: Coverage of the credible set

### Step 5: Visualize

Generate visualizations:

```python
from visualization.locus_zoom import create_locus_zoom
from visualization.pip_plot import create_pip_barplot
from visualization.interactive_report import generate_html_report
import pandas as pd

results = pd.read_csv("results/susie_results.csv")

# Static plots
create_locus_zoom(results, output_path="results/locus_zoom.png")
create_pip_barplot(results, output_path="results/pip_plot.png")

# Interactive report
generate_html_report(results, output_path="results/report.html")
```

## Example 2: Using Claude for Guided Analysis

### For Newcomers

```
"I have GWAS results for a blood pressure locus on chromosome 12.
I'm new to fine-mapping - can you help me identify the causal variant
and explain what the results mean?"
```

Claude will:
1. Guide you through preparing input files
2. Run SuSiE with appropriate parameters
3. Generate visualizations
4. Provide detailed interpretation of results

### For Experts

```
"Run SuSiE-RSS on my summary stats with EUR 1000G LD.
Use L=5, 99% coverage. Output figures as PDF."
```

Claude will:
1. Execute the analysis with specified parameters
2. Provide concise results summary
3. Generate requested outputs

## Interpreting Your Results

### Understanding PIP

**PIP (Posterior Inclusion Probability)** quantifies a variant's likelihood of being the true causal variant.

| PIP Range | Interpretation |
|-----------|----------------|
| > 0.9 | Strong evidence - likely causal |
| 0.5 - 0.9 | Moderate evidence |
| 0.1 - 0.5 | Weak evidence |
| < 0.1 | Unlikely to be causal |

### Understanding Credible Sets

A **95% credible set** contains variants that together have 95% probability of including the causal variant.

- **Single variant CS**: Strong localization, variant is likely causal
- **Large CS (>10 variants)**: Poor localization, high LD in region
- **Multiple CS**: Multiple independent causal signals

### Common Scenarios

**Scenario 1: One high-PIP variant**
```
rs12345: PIP = 0.95, CS = 1
```
Strong evidence rs12345 is the causal variant.

**Scenario 2: Multiple variants sharing signal**
```
rs12345: PIP = 0.45, CS = 1
rs12346: PIP = 0.35, CS = 1
rs12347: PIP = 0.15, CS = 1
```
Signal spread across correlated variants. All three are in the credible set.

**Scenario 3: Two independent signals**
```
rs12345: PIP = 0.90, CS = 1
rs23456: PIP = 0.85, CS = 2
```
Two independent causal signals at this locus.

## Troubleshooting

### No Credible Sets Identified

Possible causes:
1. No strong causal signal
2. L parameter too low
3. LD matrix issues

Try:
- Increase L (e.g., L=15 or L=20)
- Check LD matrix matches your variants
- Verify LD matrix is from matching population

### Large Credible Sets

Indicates high LD in the region. Consider:
- Using larger sample size if available
- Functional annotation to prioritize variants
- Multi-ancestry fine-mapping

### Convergence Issues

If SuSiE doesn't converge:
- Reduce L parameter
- Check for multicollinearity in LD matrix
- Ensure LD matrix is positive semi-definite

## Next Steps

After fine-mapping, consider:
1. **Functional annotation**: Overlap with regulatory elements
2. **Colocalization**: Test if same variant drives GWAS and eQTL signals
3. **Biological validation**: Experimental follow-up of candidate variants
