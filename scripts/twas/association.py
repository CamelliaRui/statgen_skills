# scripts/twas/association.py
"""
TWAS association testing.

Computes Z-scores for gene-trait associations using
predicted expression values.
"""

from typing import TypedDict

import numpy as np
from scipy import stats


class TWASResults(TypedDict):
    """Container for TWAS results."""
    z_scores: np.ndarray
    p_values: np.ndarray


class PowerFDRMetrics(TypedDict):
    """Power and FDR metrics."""
    power: float
    fdr: float
    n_discoveries: int
    n_true_positives: int
    n_false_positives: int


def compute_twas_z(
    pred_expression: np.ndarray,
    phenotype: np.ndarray,
) -> tuple[float, float]:
    """
    Compute TWAS Z-score for a single gene.
    
    Uses correlation-based test statistic:
    Z = r * sqrt(n-2) / sqrt(1-r^2)
    
    Args:
        pred_expression: Predicted expression (n_samples,)
        phenotype: Trait phenotype (n_samples,)
        
    Returns:
        Tuple of (z_score, p_value)
    """
    n = len(pred_expression)

    # Compute correlation
    r = np.corrcoef(pred_expression, phenotype)[0, 1]

    if np.isnan(r) or np.abs(r) > 0.9999:
        # Handle degenerate cases
        return 0.0, 1.0

    # Convert to Z-score
    z = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)

    # Two-sided p-value
    p = 2 * stats.norm.sf(np.abs(z))

    return float(z), float(p)


def run_twas(
    pred_expression: np.ndarray,
    phenotype: np.ndarray,
    gene_ids: np.ndarray | None = None,
) -> TWASResults:
    """
    Run TWAS association test for multiple genes.
    
    Args:
        pred_expression: (n_samples, n_genes) predicted expression matrix
        phenotype: (n_samples,) trait phenotype
        gene_ids: Optional gene identifiers
        
    Returns:
        TWASResults with Z-scores and p-values
    """
    n_samples, n_genes = pred_expression.shape

    z_scores = np.zeros(n_genes)
    p_values = np.ones(n_genes)

    for i in range(n_genes):
        z_scores[i], p_values[i] = compute_twas_z(
            pred_expression[:, i], phenotype
        )

    return TWASResults(
        z_scores=z_scores,
        p_values=p_values,
    )


def compute_power_fdr(
    p_values: np.ndarray,
    causal_mask: np.ndarray,
    alpha: float = 0.05,
) -> PowerFDRMetrics:
    """
    Compute power and FDR from TWAS results.
    
    Args:
        p_values: Array of p-values per gene
        causal_mask: Boolean array indicating true causal genes
        alpha: Significance threshold
        
    Returns:
        PowerFDRMetrics with power, FDR, and discovery counts
    """
    # Discoveries at significance threshold
    discoveries = p_values < alpha
    n_discoveries = np.sum(discoveries)

    # True positives: causal genes that are discovered
    true_positives = discoveries & causal_mask
    n_true_positives = np.sum(true_positives)

    # False positives: non-causal genes that are discovered
    false_positives = discoveries & ~causal_mask
    n_false_positives = np.sum(false_positives)

    # Power: fraction of causal genes discovered
    n_causal = np.sum(causal_mask)
    power = n_true_positives / n_causal if n_causal > 0 else 0.0

    # FDR: fraction of discoveries that are false
    fdr = n_false_positives / n_discoveries if n_discoveries > 0 else 0.0

    return PowerFDRMetrics(
        power=power,
        fdr=fdr,
        n_discoveries=int(n_discoveries),
        n_true_positives=int(n_true_positives),
        n_false_positives=int(n_false_positives),
    )
