# scripts/twas/models/lasso.py
"""
LASSO expression prediction model.

L1 regularization produces very sparse models, selecting
only the strongest eQTL signals.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LassoCV

from .base import ExpressionModel
from . import register_model


@register_model("lasso")
class LassoModel(ExpressionModel):
    """
    LASSO regression for expression prediction.
    
    Uses pure L1 regularization for maximum sparsity.
    Best when expecting few true causal cis-eQTLs.
    """

    def __init__(
        self,
        n_alphas: int = 100,
        cv: int = 5,
        max_iter: int = 10000,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize LASSO model.
        
        Args:
            n_alphas: Number of alpha values to try
            cv: Cross-validation folds
            max_iter: Maximum iterations
            random_state: Random seed
        """
        super().__init__(**kwargs)
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self._model: LassoCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoModel":
        """Fit LASSO with cross-validated alpha selection."""
        self._model = LassoCV(
            n_alphas=self.n_alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self.weights_ = self._model.coef_
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression values."""
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def _get_params(self) -> dict[str, Any]:
        """Get parameters for cloning."""
        return {
            "n_alphas": self.n_alphas,
            "cv": self.cv,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
