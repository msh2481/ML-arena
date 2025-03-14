from typing import Any

import numpy as np
import pandas as pd
import torch
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
from scipy.stats import norm as gaussian, t as student_t
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state


class KernelKNN(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-3, scale=0.2):
        self.alpha = alpha
        self.scale = scale

    @typed
    def fit(self, X: Float[ND, "n d_in"] | pd.DataFrame, y: Float[ND, "n"] | pd.Series):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        assert X.ndim == 2
        assert y.ndim == 1
        self.X_train = X
        self.y_train = y
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | pd.DataFrame) -> Float[ND, "m"]:
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        # Calculate RBF kernel between X and training data
        dists = (X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2
        dists = dists.sum(axis=2) / self.scale**2
        dists -= dists.min(axis=1, keepdims=True)
        K = np.exp(-dists / 2)
        assert K.shape == (X.shape[0], self.X_train.shape[0])

        # Apply regularization
        K_sum = K.sum(axis=1) + self.alpha
        assert K_sum.shape == (X.shape[0],)
        y_pred_raw = K @ self.y_train
        assert y_pred_raw.shape == (X.shape[0],)
        y_pred = y_pred_raw / K_sum

        return y_pred
