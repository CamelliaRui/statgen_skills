# scripts/twas/models/gblup.py
"""
Genomic BLUP (Best Linear Unbiased Prediction) model.

Ridge-like regression that uses all SNPs with shrinkage.
Appropriate when many SNPs have small effects (polygenic architecture).
"""

from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV

from .base import ExpressionModel
from . import register_model


@register_model("gblup")
class GBLUPModel(ExpressionModel):
    """
    GBLUP-style model using Ridge regression.
    
    Shrinks all coefficients towards zero but keeps them non-zero.
    Best for highly polygenic expression (many cis-eQTLs with small effects).
    """

    def __init__(
        self,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
        cv: int = 5,
        **kwargs: Any,
    ):
        """
        Initialize GBLUP model.
        
        Args:
            alphas: Regularization strengths to try
            cv: Cross-validation folds
        """
        super().__init__(**kwargs)
        self.alphas = alphas
        self.cv = cv
        self._model: RidgeCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBLUPModel":
        """Fit Ridge regression with cross-validated alpha."""
        self._model = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
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
            "alphas": self.alphas,
            "cv": self.cv,
        }
