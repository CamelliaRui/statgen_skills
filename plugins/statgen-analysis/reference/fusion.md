# FUSION TWAS

Run Transcriptome-Wide Association Studies using FUSION to identify genes whose expression is associated with complex traits.

## Capabilities

- Association testing with pre-computed GTEx v8 weights (49 tissues)
- Custom expression weight computation from eQTL data
- Conditional analysis to identify independent signals
- Multi-chromosome parallel execution
- Automatic reference data download and caching

## API Functions

- `run_twas_association(sumstats, tissue, output_dir)` - Run TWAS
- `list_available_tissues()` - List GTEx v8 tissues
- `download_weights(tissue)` - Download expression weights
- `check_dependencies()` - Check R, FUSION, PLINK availability

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tissue` | required | GTEx tissue name |
| `population` | EUR | LD reference population |
| `chromosomes` | 1-22 | Chromosomes to analyze |
| `gwas_n` | None | Sample size (if not in file) |
| `coloc` | False | Run colocalization |

## Requirements

- R 4.0+ with packages: optparse, glmnet
- PLINK (for custom weights only)

## Scripts

- `scripts/fusion/run_fusion.py` - Main FUSION runner
- `scripts/fusion/parsers.py` - Result parsing
- `scripts/fusion/reference_data.py` - Weight/LD reference management
- `scripts/fusion/utils.py` - Dependencies, downloads, formatting

## Workflow

1. **Check dependencies**: `check_dependencies()` — verify R, FUSION, and PLINK are available
2. **Select tissue**: `list_available_tissues()` to see available GTEx v8 tissues
3. **Download weights**: `download_weights(tissue)` — cached after first download
4. **Run TWAS**: `run_twas_association(sumstats, tissue, output_dir)`
5. **Review results**: Check significant genes (Bonferroni threshold = 0.05 / n_genes_tested)
6. **Conditional analysis** (optional): For loci with multiple significant genes, run conditional analysis to identify independent signals
7. **Visualize**: `visualization/fusion_plots.py` for result plots
