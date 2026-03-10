# scripts/twas/expression.py
"""
Expression simulation for TWAS.

Generates gene expression phenotypes from genotypes with
configurable genetic architecture (number of causal variants,
cis-heritability).
"""

from typing import TypedDict

import numpy as np


class MultiGeneResult(TypedDict):
    """Result container for multi-gene expression simulation."""
    expression: np.ndarray      # (n_samples, n_genes)
    effects: list[np.ndarray]   # List of effect arrays per gene
    causal_indices: list[np.ndarray]  # Causal SNP indices per gene


def simulate_causal_effects(
    n_snps: int,
    n_causal: int,
    h2_cis: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate causal eQTL effect sizes.
    
    Effects are scaled so that genetic variance equals h2_cis
    when genotypes are standardized.
    
    Args:
        n_snps: Number of SNPs in cis-region
        n_causal: Number of causal variants
        h2_cis: Target cis-heritability
        seed: Random seed
        
    Returns:
        Tuple of (effects array, indices of causal SNPs)
    """
    rng = np.random.default_rng(seed)

    # Sample causal indices
    if n_causal > n_snps:
        raise ValueError(f"n_causal ({n_causal}) > n_snps ({n_snps})")

    causal_idx = rng.choice(n_snps, size=n_causal, replace=False)

    # Generate raw effects from standard normal
    raw_effects = rng.standard_normal(n_causal)

    # Scale effects so sum of squared effects equals h2_cis
    # (assuming standardized genotypes with var=1)
    scale = np.sqrt(h2_cis / np.sum(raw_effects ** 2))

    effects = np.zeros(n_snps)
    effects[causal_idx] = raw_effects * scale

    return effects, causal_idx


def simulate_expression(
    genotypes: np.ndarray,
    h2_cis: float,
    n_causal: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate gene expression from genotypes.
    
    Expression = genetic component + environmental noise
    where genetic variance / total variance = h2_cis
    
    Args:
        genotypes: (n_samples, n_snps) genotype matrix (should be standardized)
        h2_cis: Cis-heritability
        n_causal: Number of causal cis-eQTLs
        seed: Random seed
        
    Returns:
        Tuple of (expression, effects, causal_indices)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_snps = genotypes.shape

    # Generate causal effects
    effects, causal_idx = simulate_causal_effects(
        n_snps=n_snps,
        n_causal=n_causal,
        h2_cis=h2_cis,
        seed=seed,
    )

    # Genetic component
    g = genotypes @ effects

    # Scale genetic component to have variance = h2_cis
    g_var = np.var(g)
    if g_var > 0:
        g = g * np.sqrt(h2_cis / g_var)

    # Environmental noise variance = 1 - h2_cis
    env_var = 1 - h2_cis
    noise = rng.standard_normal(n_samples) * np.sqrt(env_var)

    # Total expression
    expression = g + noise

    # Standardize to mean=0, var=1
    expression = (expression - expression.mean()) / (expression.std() + 1e-8)

    return expression, effects, causal_idx


def simulate_multi_gene_expression(
    genotypes_list: list[np.ndarray],
    h2_cis: float | list[float],
    n_causal: int | list[int] = 1,
    seed: int | None = None,
) -> MultiGeneResult:
    """
    Simulate expression for multiple genes.
    
    Args:
        genotypes_list: List of genotype matrices, one per gene
        h2_cis: Cis-heritability (single value or per-gene list)
        n_causal: Number of causal eQTLs (single value or per-gene list)
        seed: Random seed
        
    Returns:
        MultiGeneResult with expression matrix and effect details
    """
    rng = np.random.default_rng(seed)
    n_genes = len(genotypes_list)
    n_samples = genotypes_list[0].shape[0]

    # Convert scalar parameters to lists
    if isinstance(h2_cis, (int, float)):
        h2_cis_list = [h2_cis] * n_genes
    else:
        h2_cis_list = list(h2_cis)

    if isinstance(n_causal, int):
        n_causal_list = [n_causal] * n_genes
    else:
        n_causal_list = list(n_causal)

    # Generate seeds for each gene
    gene_seeds = rng.integers(0, 2**31, size=n_genes)

    expression_matrix = np.zeros((n_samples, n_genes))
    effects_list = []
    causal_indices_list = []

    for i in range(n_genes):
        expr, effects, causal_idx = simulate_expression(
            genotypes=genotypes_list[i],
            h2_cis=h2_cis_list[i],
            n_causal=n_causal_list[i],
            seed=int(gene_seeds[i]),
        )
        expression_matrix[:, i] = expr
        effects_list.append(effects)
        causal_indices_list.append(causal_idx)

    return MultiGeneResult(
        expression=expression_matrix,
        effects=effects_list,
        causal_indices=causal_indices_list,
    )
