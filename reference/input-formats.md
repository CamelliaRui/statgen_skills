# Input Formats

## Summary Statistics

Provide a CSV/TSV file with columns:
- `SNP` - Variant identifier (rsID)
- `CHR` - Chromosome
- `BP` - Base pair position
- `A1` - Effect allele
- `A2` - Other allele
- `BETA` and `SE` - Effect size and standard error, OR
- `Z` - Z-score
- `P` - P-value (optional, for visualization)

## LD Matrix

Three options:
1. **User-provided**: Supply a pre-computed LD matrix (numpy .npy or text)
2. **Reference panel**: Specify population (e.g., "1000G_EUR", "1000G_EAS")
3. **Compute from data**: Automatically computed when individual-level data provided

## Individual-Level Data

- PLINK format: `.bed/.bim/.fam` files
- VCF format: `.vcf.gz` with separate phenotype file
- Phenotype file: Tab-delimited with FID, IID, PHENO columns

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
