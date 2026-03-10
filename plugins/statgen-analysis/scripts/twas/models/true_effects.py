# scripts/twas/models/true_effects.py
"""
True effects oracle model.

Uses the actual causal weights for prediction.
Provides upper bound on TWAS power - what we'd achieve
with perfect expression prediction.
"""

from typing import Any

import numpy as np

from .base import ExpressionModel
from . import register_model


@register_model("true_effects")
class TrueEffectsModel(ExpressionModel):
    """
    Oracle model using true causal weights.
    
    This model doesn't learn from data - it uses pre-specified
    true causal effect sizes. Useful for:
    - Upper bound on achievable TWAS power
    - Isolating the impact of prediction error
    - Validation and debugging
    """

    def __init__(
        self,
        true_weights: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """
        Initialize with true causal weights.
        
        Args:
            true_weights: Array of true effect sizes (n_snps,)
        """
        super().__init__(**kwargs)
        self._true_weights = true_weights
        if true_weights is not None:
            self.weights_ = np.array(true_weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrueEffectsModel":
        """
        Fit is a no-op - weights are pre-specified.
        
        Validates that weights match X dimensions.
        """
        if self._true_weights is None:
            raise ValueError(
                "TrueEffectsModel requires true_weights to be set. "
                "Pass true_weights=... during initialization."
            )

        if len(self._true_weights) != X.shape[1]:
            raise ValueError(
                f"true_weights length ({len(self._true_weights)}) "
                f"doesn't match n_snps ({X.shape[1]})"
            )

        self.weights_ = np.array(self._true_weights)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using true weights."""
        if self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return X @ self.weights_

    def _get_params(self) -> dict[str, Any]:
        """Get parameters for cloning."""
        return {"true_weights": self._true_weights}
