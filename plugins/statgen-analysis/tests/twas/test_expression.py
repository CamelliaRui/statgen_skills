# tests/twas/test_expression.py
"""Tests for expression simulation."""

import pytest
import numpy as np


def test_simulate_causal_effects():
    """Test generating causal eQTL effects."""
    from scripts.twas.expression import simulate_causal_effects

    n_snps = 100
    n_causal = 5
    h2_cis = 0.1

    effects, causal_idx = simulate_causal_effects(
        n_snps=n_snps,
        n_causal=n_causal,
        h2_cis=h2_cis,
        seed=42,
    )

    assert effects.shape == (n_snps,)
    assert len(causal_idx) == n_causal
    assert np.sum(effects != 0) == n_causal


def test_simulate_expression():
    """Test simulating expression from genotypes."""
    from scripts.twas.expression import simulate_expression

    np.random.seed(42)
    n_samples = 500
    n_snps = 100
    h2_cis = 0.1

    genotypes = np.random.randint(0, 3, (n_samples, n_snps)).astype(float)
    # Standardize genotypes
    genotypes = (genotypes - genotypes.mean(0)) / (genotypes.std(0) + 1e-8)

    expression, effects, causal_idx = simulate_expression(
        genotypes=genotypes,
        h2_cis=h2_cis,
        n_causal=5,
        seed=42,
    )

    assert expression.shape == (n_samples,)
    assert effects.shape == (n_snps,)

    # Check variance is approximately as expected
    genetic_var = np.var(genotypes @ effects)
    total_var = np.var(expression)
    observed_h2 = genetic_var / total_var
    # Allow some variance due to finite sample
    assert 0.01 < observed_h2 < 0.5


def test_simulate_multi_gene_expression():
    """Test simulating expression for multiple genes."""
    from scripts.twas.expression import simulate_multi_gene_expression

    np.random.seed(42)
    n_samples = 200
    n_genes = 10
    n_snps_per_gene = 50

    # Mock genotype data per gene
    genotypes_list = [
        np.random.randn(n_samples, n_snps_per_gene)
        for _ in range(n_genes)
    ]

    result = simulate_multi_gene_expression(
        genotypes_list=genotypes_list,
        h2_cis=0.1,
        n_causal=3,
        seed=42,
    )

    assert "expression" in result
    assert "effects" in result
    assert "causal_indices" in result
    assert result["expression"].shape == (n_samples, n_genes)
    assert len(result["effects"]) == n_genes
