import numpy as np
from beartype.typing import Literal
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from wrappers import BFS, Iso, RFS, RobustRegressor, wrap


players = {
    # Classical methods
    "LinearRegression": lambda x, _: wrap(LinearRegression()),
    "Ridge": lambda x, _: wrap(Ridge(1 / (2 * len(x)))),
    "BayessianRidge": lambda x, _: wrap(BayesianRidge()),
    "ARDRegression": lambda x, _: wrap(ARDRegression()),
    # Modifications
    "Ridge": lambda x, _: wrap(Ridge(1 / (2 * len(x))), feats="id"),
    "BFS(Ridge)": lambda x, _: wrap(Ridge(1 / (2 * len(x))), feats="bfs"),
    "BFS(ARDRegression)": lambda x, _: wrap(ARDRegression(), feats="bfs"),
    # Isotonic methods
    "IsoLinearRegression": lambda x, _: wrap(LinearRegression(), isotonic=True),
    "IsoRidge": lambda x, _: wrap(Ridge(1 / (2 * len(x))), isotonic=True),
    "IsoBayessianRidge": lambda x, _: wrap(BayesianRidge(), isotonic=True),
    "IsoARDRegression": lambda x, _: wrap(ARDRegression(), isotonic=True),
    "IsoRidge": lambda x, _: wrap(Ridge(1 / (2 * len(x))), feats="id", isotonic=True),
    "IsoBFS(Ridge)": lambda x, _: wrap(
        Ridge(1 / (2 * len(x))), feats="bfs", isotonic=True
    ),
    "IsoBFS(ARDRegression)": lambda x, _: wrap(
        ARDRegression(), feats="bfs", isotonic=True
    ),
}
