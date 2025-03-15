import numpy as np
import torch
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from numpy import ndarray as ND
from sklearn.base import BaseEstimator, RegressorMixin
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


class MISO(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        feats: Literal["id", "bfs", "rfs"] = "bfs",
        final_isotonic: bool = True,
    ):
        self.isotonics = []
        self.final = wrap(
            ARDRegression(),
            feats=feats,
            isotonic=final_isotonic,
        )

    @typed
    def fit(self, X: Float[ND, "n_samples n_features"], y: Float[ND, "n_samples"]):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        self.X_train_ = X
        self.y_train_ = y
        n_samples, n_features = X.shape
        meta_features = np.zeros((n_samples, n_features))
        for i in range(n_features):
            regressor = IsotonicRegression(increasing="auto", out_of_bounds="clip")
            x_train = X[:, [i]]
            regressor.fit(x_train, y)
            self.isotonics.append(regressor)
            meta_features[:, i] = regressor.predict(x_train)
        self.final.fit(meta_features, y)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        n_samples, n_features = X.shape
        meta_features = np.zeros((n_samples, n_features))
        for i in range(n_features):
            meta_features[:, i] = self.isotonics[i].predict(X[:, [i]])
        return self.final.predict(meta_features)

    @typed
    def final_coef(self) -> Float[ND, "n_features"]:
        model = self.final
        print(type(model))
        if isinstance(model, Iso):
            model = model.estimator
        print(type(model))
        if isinstance(model, BFS):
            final = model.meta_estimator_[1].coef_
            coefs = [e[1].coef_ for e in model.estimators_]
            return sum(final[i] * coefs[i] for i in range(len(coefs)))
        print(type(model))
        if isinstance(model, Pipeline):
            model = model[-1]
        print(type(model))
        return model.coef_

    @typed
    def debug(self):
        plt.subplot(2, 1, 1)
        for i in range(len(self.isotonics)):
            x = np.sort(self.X_train_[:, [i]], axis=0)
            y = self.isotonics[i].predict(x)
            plt.plot(x, y, label=f"Isotonic {i}")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.final_coef(), label="Final")
        plt.legend()
        plt.show()


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


# # %%
# from matplotlib import pyplot as plt


# def f(X):
#     return np.log1p(X[:, 0]) + np.log1p(X[:, 1])


# N = 1000
# X = np.exp(np.random.randn(N, 2))
# y = f(X)

# model = MISO(final_isotonic=True, feats="bfs")
# model.fit(X, y)

# # Plot 2d heatmap of the predictions
# ## Create a grid of points
# x1 = np.linspace(0, 10, 100)
# x2 = np.linspace(0, 10, 100)
# X1, X2 = np.meshgrid(x1, x2)
# X_grid = np.c_[X1.ravel(), X2.ravel()]

# pred = model.predict(X_grid)
# gt = f(X_grid)

# # Plot the predictions
# plt.imshow(pred.reshape(100, 100))
# plt.colorbar()
# plt.show()

# plt.scatter(pred, gt)
# plt.plot(sorted(pred), sorted(pred))
# plt.show()

# model.debug()

# train_pred = model.predict(X)
# train_mse = np.mean((train_pred - y) ** 2)
# print(f"Train MSE: {train_mse:.4f}")
# X_test = np.exp(np.random.randn(1000, 2))
# test_pred = model.predict(X_test)
# test_mse = np.mean((test_pred - f(X_test)) ** 2)
# print(f"Test MSE: {test_mse:.4f}")


# # %%
