"""
TWAS simulation module.

Provides tools for simulating Transcriptome-Wide Association Studies,
including expression simulation, model training, and association testing.
"""

from .simulate import simulate_twas
from .expression import (
    simulate_causal_effects,
    simulate_expression,
    simulate_multi_gene_expression,
)
from .association import run_twas, compute_power_fdr
from .models import get_model, get_available_models

__all__ = [
    "simulate_twas",
    "simulate_causal_effects",
    "simulate_expression",
    "simulate_multi_gene_expression",
    "run_twas",
    "compute_power_fdr",
    "get_model",
    "get_available_models",
]
