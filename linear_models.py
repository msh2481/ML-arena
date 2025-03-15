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


class MISO(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        feats: Literal["id", "bfs", "rfs"] = "bfs",
        add_deviations: bool = False,
        final_isotonic: bool = True,
    ):
        super().__init__()
        self.isotonics = []
        self.feats = feats
        self.add_deviations = add_deviations
        self.final_isotonic = final_isotonic
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
        meta_features = np.zeros((n_samples, n_features * (1 + self.add_deviations)))
        for i in range(n_features):
            regressor = IsotonicRegression(increasing="auto", out_of_bounds="clip")
            x_train = X[:, [i]]
            regressor.fit(x_train, y)
            self.isotonics.append(regressor)
            meta_features[:, i] = regressor.predict(x_train)
        if self.add_deviations:
            self.medians = np.median(X, axis=0)
            deviations = np.abs(X - self.medians)
            for i in range(n_features):
                x_train = deviations[:, [i]]
                regressor = IsotonicRegression(increasing="auto", out_of_bounds="clip")
                regressor.fit(x_train, y)
                self.isotonics.append(regressor)
                meta_features[:, n_features + i] = regressor.predict(x_train)
        self.final.fit(meta_features, y)
        return self

    @typed
    def predict(self, X: Float[ND, "n_samples n_features"]) -> Float[ND, "n_samples"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        n_samples, n_features = X.shape
        meta_features = np.zeros((n_samples, n_features * (1 + self.add_deviations)))
        for i in range(n_features):
            meta_features[:, i] = self.isotonics[i].predict(X[:, [i]])
        if self.add_deviations:
            deviations = np.abs(X - self.medians)
            for i in range(n_features):
                x_train = deviations[:, [i]]
                meta_features[:, n_features + i] = self.isotonics[
                    n_features + i
                ].predict(x_train)
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


def test_miso_is_regressor():
    from sklearn.base import is_regressor

    assert is_regressor(MISO())


if __name__ == "__main__":
    test_miso_is_regressor()
