# TWAS Simulator Design

**Date:** 2026-01-18
**Status:** Design Complete

## Overview

Native Python implementation of TWAS simulation, ported from [mancusolab/twas_sim](https://github.com/mancusolab/twas_sim), providing simulation capabilities for methods development, power analysis, and teaching.

## Design Decisions

- **Use case:** All (methods development, power analysis, teaching)
- **Genotype data:** Hybrid approach (user-provided PLINK + optional 1000G download)
- **Integration:** Native port (port core logic into statgen_skills)
- **Expression models:** Full parity (Elastic Net, LASSO, GBLUP, true effects, external)

## Architecture

### Core Components

```
scripts/twas/
├── __init__.py              # Public API exports
├── simulate.py              # Main simulation functions
├── genotype.py              # Genotype loading & 1000G download
├── expression.py            # Expression simulation
├── association.py           # TWAS association testing
├── power.py                 # Power analysis & batch modes
└── models/
    ├── __init__.py          # Model registry
    ├── base.py              # Abstract base class
    ├── elastic_net.py
    ├── lasso.py
    ├── gblup.py
    ├── true_effects.py
    └── external.py

visualization/
└── twas_plots.py            # All TWAS visualizations

tests/twas/
├── test_genotype.py
├── test_expression.py
├── test_models.py
├── test_simulate.py
└── test_integration.py

examples/
├── twas_sim_example.py      # Quick start script
└── twas_tutorial.md         # Tutorial with explanations
```

## Simulation Parameters

### Genetic Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_causal_cis` | 1 | Number of causal cis-eQTLs per gene |
| `h2_cis` | 0.1 | Cis-heritability of expression |
| `h2_trait` | 0.5 | Total trait heritability |
| `prop_mediated` | 0.1 | Proportion of h²_trait mediated through expression |
| `n_genes` | 100 | Number of genes to simulate |
| `n_causal_genes` | 10 | Genes with true trait effects |

### Sample Sizes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gwas` | 50000 | GWAS sample size |
| `n_eqtl` | 500 | eQTL training sample size |
| `n_test` | 1000 | Held-out test set for model evaluation |

### Simulation Modes

1. **Single run** - One simulation with full outputs
2. **Power analysis** - Multiple replicates varying one parameter
3. **Batch mode** - Grid search across parameter combinations

## Expression Prediction Models

### Base Interface

```python
class ExpressionModel(ABC):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExpressionModel"
    def predict(self, X: np.ndarray) -> np.ndarray
    def get_weights(self) -> np.ndarray
    def get_nonzero_snps(self) -> List[int]
    def cross_validate(self, X, y, cv=5) -> dict
```

### Model Implementations

| Model | Description | Use Case |
|-------|-------------|----------|
| **Elastic Net** | L1+L2 penalty, sparse | Default, realistic TWAS |
| **LASSO** | L1 only, very sparse | When expecting few eQTLs |
| **GBLUP** | Ridge-like, all SNPs | Dense genetic architecture |
| **True Effects** | Oracle weights | Upper bound on power |
| **External** | PrediXcan/FUSION format | Validate against real models |

## API

### Main Entry Points

```python
from scripts.twas import simulate_twas, power_analysis, batch_simulate

# Single simulation
result = simulate_twas(
    genotype="path/to/plink_prefix",  # or "1000G_EUR_chr22"
    n_genes=100,
    n_causal_genes=10,
    h2_cis=0.1,
    h2_trait=0.5,
    prop_mediated=0.1,
    n_gwas=50000,
    n_eqtl=500,
    models=["elastic_net", "lasso", "true_effects"],
    output_dir="results/",
    seed=42
)

# Power analysis (vary one parameter)
power = power_analysis(
    genotype="path/to/plink_prefix",
    vary_param="n_eqtl",
    vary_values=[100, 250, 500, 1000],
    n_replicates=100,
    output_dir="power_results/"
)

# Batch mode (grid search)
batch_simulate(
    genotype="path/to/plink_prefix",
    param_grid={
        "h2_cis": [0.05, 0.1, 0.2],
        "n_eqtl": [250, 500, 1000]
    },
    n_replicates=50,
    output_dir="batch_results/"
)
```

### Simulation Pipeline (internal)

1. Load/download genotypes → subset to cis-region per gene
2. Sample causal eQTL effects → simulate expression
3. Split samples: train (n_eqtl) / test
4. Train expression models on training set
5. Evaluate prediction on test set
6. Generate GWAS phenotype (expression-mediated + direct)
7. Run TWAS association (predicted expression ~ trait)
8. Compute metrics: power, FDR, calibration

## Outputs

### Per Simulation

| File | Description |
|------|-------------|
| `simulation_params.json` | All input parameters and random seed |
| `true_effects.csv` | Ground truth: causal genes, SNPs, effect sizes |
| `expression_matrix.csv` | Simulated gene expression (genes × samples) |
| `weights_{model}.csv` | Trained weights per model |
| `twas_results.csv` | Association Z-scores, p-values per gene |
| `model_performance.csv` | R², correlation, n_nonzero per model |
| `summary.json` | Power, FDR, calibration metrics |

### Power Analysis

| File | Description |
|------|-------------|
| `power_curve.csv` | Power at each parameter value |
| `power_curve.png` | Line plot of power vs parameter |
| `fdr_curve.png` | FDR calibration across thresholds |
| `metrics_summary.csv` | Aggregated metrics across replicates |

## Visualizations

```python
# Power analysis
create_power_curve(results, vary_param, output)
create_fdr_calibration(results, output)

# Model comparison
create_model_comparison(results, output)
create_weight_correlation(weights_dict, output)

# Single simulation
create_twas_manhattan(results, output)
create_qq_plot(results, output)
```

## Dependencies

```
scikit-learn>=1.0    # ElasticNet, LASSO, cross-validation
pandas-plink>=2.0    # PLINK file reading
```

## Implementation Order

1. `genotype.py` - Load PLINK, optional 1000G download
2. `models/` - All expression prediction models
3. `expression.py` - Simulate expression from genotypes
4. `association.py` - TWAS Z-score computation
5. `simulate.py` - Main orchestration
6. `power.py` - Power analysis modes
7. `visualization/twas_plots.py` - Plotting functions
8. Tests and documentation
