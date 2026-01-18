# scripts/twas/models/__init__.py
"""
Expression prediction models for TWAS simulation.

Available models:
- elastic_net: ElasticNet regression (L1 + L2)
- lasso: LASSO regression (L1 only)
- gblup: Genomic BLUP (ridge-like)
- true_effects: Oracle model with true causal weights
- external: Load pre-trained weights
"""

from typing import Type

from .base import ExpressionModel

# Model registry - populated as models are implemented
_MODEL_REGISTRY: dict[str, Type[ExpressionModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type[ExpressionModel]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_available_models() -> list[str]:
    """Return list of available model names."""
    # Return all planned models even if not yet implemented
    return ["elastic_net", "lasso", "gblup", "true_effects", "external"]


def get_model(name: str, **kwargs) -> ExpressionModel:
    """
    Get a model instance by name.
    
    Args:
        name: Model name (elastic_net, lasso, gblup, true_effects, external)
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name not recognized
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        if not available:
            raise ValueError(
                f"Unknown model: {name}. No models registered yet."
            )
        raise ValueError(
            f"Unknown model: {name}. Available: {available}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "ExpressionModel",
    "get_available_models",
    "get_model",
    "register_model",
]

# Import models to trigger registration
from . import elastic_net
from . import lasso
