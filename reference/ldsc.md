# LDSC (LD Score Regression)

Estimate SNP heritability, genetic correlations, and partition heritability by functional annotations.

## Capabilities

- SNP heritability (h²) estimation from GWAS summary statistics
- Genetic correlation (rg) between multiple traits
- Stratified LDSC (s-LDSC) for heritability partitioning by annotations
- Automatic reference data download for EUR, EAS, AFR, SAS, AMR populations
- Publication-ready visualization (h² bar charts, rg heatmaps, enrichment plots)

## API Functions

- `estimate_heritability(sumstats, output_dir, population)` - SNP h² from GWAS
- `genetic_correlation(sumstats_list, output_dir, population)` - rg between traits
- `partitioned_heritability(sumstats, annotations, output_dir)` - s-LDSC enrichment
- `munge_sumstats(input, output)` - Convert summary stats to LDSC format

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population` | EUR | Reference population (EUR, EAS, AFR, SAS, AMR) |
| `no_intercept` | False | Constrain intercept to 1 |
| `samp_prev` | None | Sample prevalence (case-control) |
| `pop_prev` | None | Population prevalence (case-control) |

## Scripts

- `scripts/ldsc/run_ldsc.py` - Main entry point for all LDSC analyses
- `scripts/ldsc/munge.py` - Summary statistics preprocessing
- `scripts/ldsc/reference_data.py` - Reference data management
- `scripts/ldsc/parsers.py` - Log file parsing

## Workflow

1. **Munge summary stats**: `munge_sumstats(input, output)`
   - Verify output SNP count is reasonable (warn if <500K for well-powered GWAS)
   - Check for column mapping warnings
2. **Run analysis**:
   - Heritability: `estimate_heritability(sumstats, output_dir, population)`
   - Genetic correlation: `genetic_correlation(sumstats_list, output_dir, population)`
   - Partitioned: `partitioned_heritability(sumstats, annotations, output_dir)`
3. **Check intercept**: If intercept > 1.1, warn about possible population stratification or sample overlap
4. **Review results**: h² should be between 0 and 1; rg between -1 and 1. Out-of-range values suggest data issues
5. **Visualize**:
   - `visualization/ldsc_plots.py` for h² bar charts, rg heatmaps, enrichment plots
6. **Build report**: Generate interactive HTML with `visualization/interactive_report.py`
