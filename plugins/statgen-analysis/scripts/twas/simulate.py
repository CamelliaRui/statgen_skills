# scripts/twas/simulate.py
"""
Main TWAS simulation orchestrator.

Coordinates:
1. Expression simulation
2. Model training
3. TWAS association testing
4. Result aggregation and output
"""

import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from .expression import simulate_multi_gene_expression
from .association import run_twas, compute_power_fdr
from .models import get_model


class SimulationResult(TypedDict):
    """Complete simulation result."""
    twas_results: dict[str, dict]
    model_performance: dict[str, dict]
    true_effects: dict
    power_metrics: dict[str, dict]


def simulate_twas(
    genotypes_list: list[np.ndarray],
    n_causal_genes: int,
    h2_cis: float = 0.1,
    h2_trait: float = 0.5,
    prop_mediated: float = 0.1,
    n_causal_cis: int = 1,
    models: list[str] = ["elastic_net"],
    output_dir: str | Path | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run a complete TWAS simulation.
    
    Args:
        genotypes_list: List of genotype matrices (n_samples, n_snps) per gene
        n_causal_genes: Number of genes with true trait effects
        h2_cis: Cis-heritability of expression
        h2_trait: Total trait heritability
        prop_mediated: Proportion of h2_trait mediated through expression
        n_causal_cis: Number of causal cis-eQTLs per gene
        models: List of model names to use
        output_dir: Directory to save outputs
        seed: Random seed
        verbose: Print progress
        
    Returns:
        SimulationResult with all results
    """
    rng = np.random.default_rng(seed)
    n_genes = len(genotypes_list)
    n_samples = genotypes_list[0].shape[0]

    if verbose:
        print(f"Running TWAS simulation: {n_genes} genes, {n_samples} samples")

    # Setup output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simulate expression
    if verbose:
        print("Simulating gene expression...")

    expr_result = simulate_multi_gene_expression(
        genotypes_list=genotypes_list,
        h2_cis=h2_cis,
        n_causal=n_causal_cis,
        seed=seed,
    )
    expression = expr_result["expression"]

    # 2. Select causal genes for trait
    causal_gene_idx = rng.choice(n_genes, size=n_causal_genes, replace=False)
    causal_mask = np.zeros(n_genes, dtype=bool)
    causal_mask[causal_gene_idx] = True

    # 3. Simulate trait phenotype
    # Variance mediated through expression
    mediated_var = h2_trait * prop_mediated
    # Direct genetic variance (not through expression)
    direct_var = h2_trait * (1 - prop_mediated)
    # Environmental variance
    env_var = 1 - h2_trait

    # Gene effects on trait
    gene_effects = np.zeros(n_genes)
    gene_effects[causal_gene_idx] = rng.standard_normal(n_causal_genes)
    gene_effects = gene_effects / (np.sqrt(np.sum(gene_effects**2)) + 1e-8)
    gene_effects *= np.sqrt(mediated_var)

    # Phenotype components
    mediated_component = expression @ gene_effects
    direct_component = rng.standard_normal(n_samples) * np.sqrt(direct_var)
    env_component = rng.standard_normal(n_samples) * np.sqrt(env_var)

    phenotype = mediated_component + direct_component + env_component
    phenotype = (phenotype - phenotype.mean()) / (phenotype.std() + 1e-8)

    # 4. Split samples: eQTL training vs GWAS
    n_eqtl = min(n_samples // 2, 500)
    n_gwas = n_samples - n_eqtl

    all_idx = rng.permutation(n_samples)
    eqtl_idx = all_idx[:n_eqtl]
    gwas_idx = all_idx[n_eqtl:]

    # 5. Train models and predict expression
    if verbose:
        print(f"Training models: {models}")

    twas_results = {}
    model_performance = {}

    for model_name in models:
        if verbose:
            print(f"  Training {model_name}...")

        pred_expression = np.zeros((n_gwas, n_genes))
        cv_metrics = []

        for g in range(n_genes):
            # Get genotypes for this gene
            X = genotypes_list[g]
            y = expression[:, g]

            # Train on eQTL samples
            X_train, y_train = X[eqtl_idx], y[eqtl_idx]
            X_test = X[gwas_idx]

            # Get model (with true weights for oracle model)
            if model_name == "true_effects":
                model = get_model(
                    model_name,
                    true_weights=expr_result["effects"][g]
                )
            else:
                model = get_model(model_name, random_state=seed)

            # Cross-validate and fit
            try:
                metrics = model.cross_validate(X_train, y_train, cv=5, seed=seed)
                cv_metrics.append(metrics)
            except Exception:
                # Fallback if CV fails
                model.fit(X_train, y_train)
                cv_metrics.append({"cv_r2": 0, "cv_corr": 0, "n_nonzero": 0})

            # Predict on GWAS samples
            pred_expression[:, g] = model.predict(X_test)

        # Run TWAS
        results = run_twas(pred_expression, phenotype[gwas_idx])

        # Compute power/FDR
        power_fdr = compute_power_fdr(
            results["p_values"], causal_mask, alpha=0.05
        )

        twas_results[model_name] = {
            "z_scores": results["z_scores"].tolist(),
            "p_values": results["p_values"].tolist(),
        }

        model_performance[model_name] = {
            "mean_cv_r2": np.mean([m["cv_r2"] for m in cv_metrics]),
            "mean_cv_corr": np.mean([m["cv_corr"] for m in cv_metrics]),
            "mean_n_nonzero": np.mean([m["n_nonzero"] for m in cv_metrics]),
        }

        twas_results[model_name]["power_metrics"] = dict(power_fdr)

    # 6. Compile results
    true_effects_info = {
        "causal_gene_indices": causal_gene_idx.tolist(),
        "gene_trait_effects": gene_effects.tolist(),
        "cis_effects": [e.tolist() for e in expr_result["effects"]],
        "cis_causal_indices": [idx.tolist() for idx in expr_result["causal_indices"]],
    }

    result = SimulationResult(
        twas_results=twas_results,
        model_performance=model_performance,
        true_effects=true_effects_info,
        power_metrics={m: twas_results[m]["power_metrics"] for m in models},
    )

    # 7. Save outputs
    if output_dir is not None:
        _save_outputs(
            output_dir=output_dir,
            result=result,
            params={
                "n_genes": n_genes,
                "n_samples": n_samples,
                "n_causal_genes": n_causal_genes,
                "h2_cis": h2_cis,
                "h2_trait": h2_trait,
                "prop_mediated": prop_mediated,
                "n_causal_cis": n_causal_cis,
                "models": models,
                "seed": seed,
            },
            verbose=verbose,
        )

    if verbose:
        print("Simulation complete!")
        for m in models:
            pm = result["power_metrics"][m]
            print(f"  {m}: power={pm['power']:.3f}, FDR={pm['fdr']:.3f}")

    return result


def _save_outputs(
    output_dir: Path,
    result: SimulationResult,
    params: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Save simulation outputs to files."""

    # Parameters
    with open(output_dir / "simulation_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # True effects
    pd.DataFrame({
        "gene": range(len(result["true_effects"]["gene_trait_effects"])),
        "trait_effect": result["true_effects"]["gene_trait_effects"],
        "is_causal": [
            i in result["true_effects"]["causal_gene_indices"]
            for i in range(len(result["true_effects"]["gene_trait_effects"]))
        ],
    }).to_csv(output_dir / "true_effects.csv", index=False)

    # TWAS results (one row per gene, columns for each model)
    n_genes = len(result["true_effects"]["gene_trait_effects"])
    twas_df = pd.DataFrame({"gene": range(n_genes)})
    for model_name, model_results in result["twas_results"].items():
        twas_df[f"{model_name}_z"] = model_results["z_scores"]
        twas_df[f"{model_name}_p"] = model_results["p_values"]
    twas_df.to_csv(output_dir / "twas_results.csv", index=False)

    # Model performance
    perf_data = []
    for model_name, perf in result["model_performance"].items():
        perf_data.append({"model": model_name, **perf})
    pd.DataFrame(perf_data).to_csv(output_dir / "model_performance.csv", index=False)

    # Summary
    summary = {
        "power_metrics": result["power_metrics"],
        "model_performance": result["model_performance"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"Outputs saved to: {output_dir}")
