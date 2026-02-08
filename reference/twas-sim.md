# TWAS Simulator

Simulate Transcriptome-Wide Association Studies for methods development, power analysis, and teaching.

## Capabilities

- Simulate gene expression with configurable cis-heritability
- Multiple expression prediction models (Elastic Net, LASSO, GBLUP, oracle)
- Full TWAS pipeline: expression → weights → association
- Power and FDR calculation
- Publication-ready visualizations (power curves, Manhattan, QQ plots)

## API Functions

- `simulate_twas(genotypes, n_causal_genes, ...)` - Run complete simulation
- `simulate_expression(genotypes, h2_cis, n_causal)` - Simulate expression only
- `run_twas(pred_expression, phenotype)` - TWAS association test
- `get_model(name)` - Get expression prediction model by name

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h2_cis` | 0.1 | Cis-heritability of expression |
| `h2_trait` | 0.5 | Total trait heritability |
| `prop_mediated` | 0.1 | Fraction of h² mediated through expression |
| `n_causal_cis` | 1 | Causal cis-eQTLs per gene |
| `n_causal_genes` | 10 | Genes with trait effects |

## Scripts

- `scripts/twas/simulate.py` - Main TWAS simulation orchestrator
- `scripts/twas/expression.py` - Expression simulation
- `scripts/twas/association.py` - TWAS association testing
- `scripts/twas/genotype.py` - Genotype loading and processing
- `scripts/twas/models/` - Expression prediction models:
  - `elastic_net.py` - ElasticNet with CV
  - `lasso.py` - LASSO regression
  - `gblup.py` - Genomic REML
  - `true_effects.py` - Oracle model (uses true causal weights)

## Models

Models are registered via `@register_model` decorator and retrieved with `get_model(name)`:
- `"elastic_net"` - Default; good bias-variance tradeoff
- `"lasso"` - Sparse selection; best when few large eQTLs
- `"gblup"` - Ridge-like; best when many small eQTLs
- `"true_effects"` - Oracle; upper bound on TWAS power

## Workflow

1. **Load genotypes**: Provide PLINK or numpy genotype matrix
2. **Configure simulation**: Set `h2_cis`, `n_causal_genes`, `prop_mediated`, and model type
3. **Run simulation**: `simulate_twas(genotypes, ...)`
4. **Check calibration**: Under the null (`n_causal_genes=0`), p-values should be uniform — inspect QQ plot
5. **Evaluate power**: Compare power across parameter settings or models
6. **Visualize**: Power curves (`visualization/twas_plots.py`), Manhattan, QQ plots
