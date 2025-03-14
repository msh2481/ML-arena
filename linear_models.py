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
from wrappers import BFS, Iso, RFS, RobustRegressor


def wrap(
    model: BaseEstimator,
    scale_kind: Literal["standard", "robust", "quantile", "id"] = "standard",
    feats: Literal["bfs", "rfs", "id"] = "id",
    outliers: Literal["robust", "id"] = "id",
    isotonic: bool = False,
) -> BaseEstimator:
    scaled_model = None
    if scale_kind == "standard":
        scaled_model = Pipeline([("scaler", StandardScaler()), ("model", model)])
    elif scale_kind == "robust":
        scaled_model = Pipeline([("scaler", RobustScaler()), ("model", model)])
    elif scale_kind == "quantile":
        scaled_model = Pipeline([("scaler", QuantileTransformer()), ("model", model)])
    elif scale_kind == "id":
        scaled_model = model
    else:
        raise ValueError(f"Invalid scaling method: {scale_kind}")
    feature_selected = None
    if feats == "bfs":
        feature_selected = BFS(scaled_model)
    elif feats == "rfs":
        feature_selected = RFS(scaled_model)
    elif feats == "id":
        feature_selected = scaled_model
    else:
        raise ValueError(f"Invalid feature selection method: {feats}")
    outlier_removed = None
    if outliers == "robust":
        outlier_removed = RobustRegressor(feature_selected)
    elif outliers == "id":
        outlier_removed = feature_selected
    else:
        raise ValueError(f"Invalid outlier removal method: {outliers}")
    final = None
    if isotonic:
        final = Iso(outlier_removed)
    else:
        final = outlier_removed
    return final


# players = {
#     "FeatureStacking(Ridge(1/(2n)))": lambda x, _: RFS(
#         scale(Ridge(alpha=1.0 / (2 * len(x))))
#     ),
#     "FeatureStacking(Lasso(1/(2n)))": lambda x, _: RFS(
#         scale(Lasso(alpha=1.0 / (2 * len(x))))
#     ),
#     "Ridge(1/(2n))": lambda x, _: scale(Ridge(alpha=1.0 / (2 * len(x)))),
#     "ARDRegression": lambda x, _: scale(ARDRegression()),
#     "LinearRegression": lambda x, _: scale(LinearRegression()),
#     "RidgeCV(cv=10)": lambda x, _: scale(
#         RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=10)
#     ),
#     "Lasso(1/(2n))": lambda x, _: scale(Lasso(alpha=1.0 / (2 * len(x)))),
#     "LassoCV(cv=10)": lambda x, _: scale(LassoCV(cv=10)),
#     # Now with robust scaling
#     "robust-ARDRegression": lambda x, _: scale(ARDRegression(), kind="robust"),
#     "robust-LinearRegression": lambda x, _: scale(LinearRegression(), kind="robust"),
#     "robust-RidgeCV(cv=10)": lambda x, _: scale(
#         RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=10), kind="robust"
#     ),
#     "robust-LassoCV(cv=10)": lambda x, _: scale(LassoCV(cv=10), kind="robust"),
# }


# models = [
#     scale(Ridge(1 / (2 * len(X_train)))),
#     scale(ARDRegression()),
#     scale(BFS(inner)),
#     scale(BFS(inner, use_positive=False)),
#     scale(BFS(inner, use_scaling=False)),
#     scale(RFS(inner)),
#     scale(RFS(inner, use_positive=False)),
# ]
# model_names = [
#     "Ridge(1/(2n))",
#     "ARDRegression",
#     "BFS(inner)",
#     "BFS(inner, use_positive=False)",
#     "BFS(inner, use_scaling=False)",
#     "RFS(inner)",
#     "RFS(inner, use_positive=False)",
# ]

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
