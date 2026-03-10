# N3 Fine-Mapping Tutorial

This tutorial demonstrates SuSiE fine-mapping using the N3finemapping dataset from the susieR package. This is the same example used in the [official susieR vignette](https://stephenslab.github.io/susieR/articles/finemapping_summary_statistics.html).

## Dataset Overview

The N3finemapping dataset is a simulated genetic fine-mapping scenario with:
- **1,001 variants** (SNPs)
- **574 samples**
- **3 true causal variants** at positions 403, 653, and 773

## Files

| File | Description |
|------|-------------|
| `N3_sumstats.csv` | Summary statistics (BETA, SE, Z, P) |
| `N3_ld.txt` | LD correlation matrix (1001 x 1001) |
| `N3_ld.rds` | LD matrix in R format |

## Running the Analysis

### Using the R Script

```bash
Rscript scripts/susie/run_susie.R \
    --input examples/N3_sumstats.csv \
    --input-type sumstats \
    --ld examples/N3_ld.txt \
    --n 574 \
    --output results/ \
    --L 10 \
    --coverage 0.95
```

### Using Claude with the Skill

```
"Run SuSiE fine-mapping on the N3 example data.
Sample size is 574. Identify the causal variants."
```

## Expected Results

SuSiE identifies **3 credible sets** that capture all 3 true causal signals:

### Credible Set 1 (SNP653)
- **Lead variant**: SNP653 (true causal)
- **PIP**: 0.998
- **Size**: 1 variant
- **Interpretation**: Strong, well-localized signal

### Credible Set 2 (SNP773, SNP777)
- **Lead variant**: SNP773 (true causal)
- **PIP**: 0.593 (SNP773), 0.404 (SNP777)
- **Size**: 2 variants
- **Interpretation**: Signal shared between two variants in LD

### Credible Set 3 (30 variants including SNP403)
- **Lead variant**: SNP381
- **True causal (SNP403) PIP**: 0.033
- **Size**: 30 variants
- **Interpretation**: Diffuse signal due to high LD in the region

## Comparison with Official Results

Our results exactly match the [susieR vignette](https://stephenslab.github.io/susieR/articles/finemapping_summary_statistics.html):

> "The three causal signals have been captured by the three CSs. Note the third CS contains many variables, including the true causal variable 403."

| Credible Set | susieR Vignette | Our Results |
|--------------|-----------------|-------------|
| CS1 | Variable 653 | SNP653 (PIP=0.998) |
| CS2 | Variables 773, 777 | SNP773 (0.593), SNP777 (0.404) |
| CS3 | 30 vars incl. 403 | 30 variants, SNP403 (0.033) |

## Interpreting the Results

### Why is SNP403 hard to identify?

The third causal variant (SNP403) has a low PIP (0.033) despite being truly causal. This happens because:

1. **High LD**: SNP403 is in a region with many correlated variants
2. **Signal spreading**: The causal effect is "spread" across correlated variants
3. **Limited sample size**: With only 574 samples, distinguishing correlated variants is challenging

This is a realistic fine-mapping scenario that demonstrates both the power and limitations of statistical fine-mapping.

### Key Takeaways

1. **Credible sets capture causal variants** - All 3 true causal variants are in credible sets
2. **PIP varies by LD structure** - High LD regions produce lower PIPs for individual variants
3. **CS size indicates resolution** - Small CS = good localization, large CS = LD-limited
4. **Lead variant may not be causal** - In CS3, SNP381 (not causal) has higher PIP than SNP403 (causal)

## Generating Visualizations

```python
import pandas as pd
from visualization.pip_plot import create_pip_barplot
from visualization.interactive_report import generate_html_report

results = pd.read_csv('results/susie_results.csv')

# Create PIP bar plot
create_pip_barplot(results, output_path='results/pip_plot.png', top_n=15)

# Generate interactive HTML report
generate_html_report(
    results,
    output_path='results/report.html',
    title='N3 Fine-Mapping Results'
)
```

## References

- [susieR package](https://github.com/stephenslab/susieR)
- [Fine-mapping vignette](https://stephenslab.github.io/susieR/articles/finemapping_summary_statistics.html)
- Wang et al. (2020). "A simple new approach to variable selection in regression, with application to genetic fine-mapping." *JRSS-B*.
