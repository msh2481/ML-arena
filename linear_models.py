import numpy as np
from sklearn.linear_model import ARDRegression, LinearRegression, Ridge, RidgeCV


players = {
    "Ridge(adaptive)": lambda x, _: Ridge(alpha=1.0 / (2 * len(x))),
    "Ridge(1.0)": lambda x, _: Ridge(alpha=1.0),
    "ARDRegression": lambda x, _: ARDRegression(),
    "LinearRegression": lambda x, _: LinearRegression(),
    "RidgeCV": lambda x, _: RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=3),
}
