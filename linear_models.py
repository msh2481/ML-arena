import numpy as np
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from wrappers import FeatureStackingRegressor, RobustRegressor


players = {
    "FeatureStacking(Ridge(1/(2n)))": lambda x, _: FeatureStackingRegressor(
        Ridge(alpha=1.0 / (2 * len(x)))
    ),
    "FeatureStacking(Lasso(1/(2n)))": lambda x, _: FeatureStackingRegressor(
        Lasso(alpha=1.0 / (2 * len(x)))
    ),
    "Ridge(1/(2n))": lambda x, _: Ridge(alpha=1.0 / (2 * len(x))),
    "ARDRegression": lambda x, _: ARDRegression(),
    "LinearRegression": lambda x, _: LinearRegression(),
    "RidgeCV(cv=10)": lambda x, _: RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=10),
    "Lasso(1/(2n))": lambda x, _: Lasso(alpha=1.0 / (2 * len(x))),
    "LassoCV(cv=10)": lambda x, _: LassoCV(cv=10),
}
