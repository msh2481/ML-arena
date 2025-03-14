import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from wrappers import BFS, RFS, RobustRegressor


def scale(model: BaseEstimator, kind: str = "standard") -> BaseEstimator:
    if kind == "standard":
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    elif kind == "robust":
        return Pipeline([("scaler", RobustScaler()), ("model", model)])
    elif kind == "id":
        return model
    else:
        raise ValueError(f"Invalid scaling method: {kind}")


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
    "Ridge(1/(2n))": lambda x, _: scale(Ridge(1 / (2 * len(x)))),
    "ARDRegression": lambda x, _: scale(ARDRegression()),
    "BFS(inner, cv=3)": lambda x, _: scale(BFS(Ridge(1 / (2 * len(x))), cv=3)),
    "BFS(inner, cv=5)": lambda x, _: scale(BFS(Ridge(1 / (2 * len(x))), cv=5)),
    "RFS(inner)": lambda x, _: scale(RFS(Ridge(1 / (2 * len(x))))),
    "LinearRegression": lambda x, _: scale(LinearRegression()),
    "RidgeCV(cv=10)": lambda x, _: scale(
        RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=10)
    ),
    "LassoCV(cv=10)": lambda x, _: scale(LassoCV(cv=10)),
}
