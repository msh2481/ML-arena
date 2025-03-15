import numpy as np
import torch
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from lightgbm import LGBMRegressor
from linear_models import MISO
from loguru import logger
from numpy import ndarray as ND
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ARDRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from wrappers import BFS, Iso, RFS, RobustRegressor, wrap
from xgboost import XGBRegressor


class Horizontal(BaseEstimator, RegressorMixin):
    def __init__(self, use_gs: bool = True):
        self.miso = MISO()
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=3)
        self.lgbm = LGBMRegressor(learning_rate=0.05, max_depth=3)
        if use_gs:
            self.lgbm = GS(self.lgbm)

    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        self.miso.fit(X, y)
        self.rf.fit(X, y)
        self.lgbm.fit(X, y)
        return self

    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        meta_features = np.zeros((X.shape[0], 3))
        meta_features[:, 0] = self.miso.predict(X)
        meta_features[:, 1] = self.rf.predict(X)
        meta_features[:, 2] = self.lgbm.predict(X)
        return (
            0.25 * meta_features[:, 0]
            + 0.25 * meta_features[:, 1]
            + 0.5 * meta_features[:, 2]
        )


class Hybrid(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        add_deviations: bool = False,
        use_gs: bool = False,
        use_isotonic: bool = False,
        tree_type: Literal["lgbm", "rf"] = "lgbm",
    ):
        self.use_gs = use_gs
        self.use_isotonic = use_isotonic
        self.miso = MISO(
            feats="id", final_isotonic=False, add_deviations=add_deviations
        )
        if tree_type == "lgbm":
            self.tree = LGBMRegressor(learning_rate=0.05, max_depth=3)
        elif tree_type == "rf":
            self.tree = RandomForestRegressor(n_estimators=100, max_depth=3)
        if self.use_gs and tree_type != "rf":
            self.tree = GS(self.tree)
        if self.use_isotonic:
            self.isotonic = IsotonicRegression(out_of_bounds="clip")

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        self.miso.fit(X, y)
        miso_pred = self.miso.predict(X)
        self.tree.fit(X, y - miso_pred)
        tree_pred = self.tree.predict(X)
        if self.use_isotonic:
            self.isotonic.fit(miso_pred + tree_pred, y)
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        miso_pred = self.miso.predict(X)
        tree_pred = self.tree.predict(X)
        if self.use_isotonic:
            isotonic_pred = self.isotonic.predict(miso_pred + tree_pred)
            return isotonic_pred
        return miso_pred + tree_pred


class GS(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: BaseEstimator,
        top_k: int = 3,
        max_depths: list[int] = [2, 3],
        learning_rates: list[float] = [0.02, 0.05, 0.1, 0.2],
        cv: int = 2,
    ):
        self.estimator = estimator
        self.top_k = top_k
        self.param_grid = {
            "max_depth": max_depths,
            "learning_rate": learning_rates,
        }
        self.cv = cv

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        self.search_ = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
        )

        self.search_.fit(X, y)
        self.best_params_ = self.search_.best_params_

        # Get top-3 models from the search results
        results = self.search_.cv_results_
        indices = np.argsort(results["mean_test_score"])[-self.top_k :]

        estimators = []
        for idx in indices:
            params = {k: results["param_" + k][idx] for k in self.param_grid.keys()}
            # logger.info(f"Best parameters #{idx}: {params}")
            estimator = clone(self.estimator)
            estimator.set_params(**params)
            estimator.fit(X, y)
            estimators.append(estimator)

        self.best_estimator_ = VotingRegressor(
            estimators=[("model" + str(i), est) for i, est in enumerate(estimators)]
        )
        self.best_estimator_.fit(X, y)

        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        return self.best_estimator_.predict(X)


def smoothed_prediction(
    estimator: BaseEstimator,
    X: Float[ND, "n d_in"],
    noise: Float[ND, "noise_samples"],
) -> Float[ND, "n"]:
    Xs = []
    for d in noise:
        Xs.append(X + d)
    Xs = np.concatenate(Xs, axis=0)
    ys = estimator.predict(Xs)
    ys = ys.reshape(len(noise), len(X))
    return np.mean(ys, axis=0)


class Smooth(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: BaseEstimator,
        noise_levels: list[float] = [0.1, 0.2, 0.5],
        noise_samples: int = 5,
        cv: int = 2,
    ):
        self.scaler = StandardScaler()
        self.estimator = estimator
        self.cv = cv
        self.noise_levels = noise_levels
        delta = 1 / (2 * noise_samples)
        self.noise_quantiles = norm.ppf(
            np.linspace(
                delta,
                1 - delta,
                noise_samples,
                endpoint=True,
            )
        )
        self.fitted_estimators_ = []

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        meta_features = np.zeros((n_samples, 1 + len(self.noise_levels)))
        kf = KFold(n_splits=self.cv, shuffle=False)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            estimator = clone(self.estimator)
            estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            self.fitted_estimators_.append(estimator)
            meta_features[test_idx, 0] = estimator.predict(X_test)
            for i, noise_level in enumerate(self.noise_levels):
                predictions = smoothed_prediction(
                    estimator, X_test, self.noise_quantiles * noise_level
                )
                meta_features[test_idx, 1 + i] = predictions
        self.meta_estimator_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "estimator",
                    Lasso(fit_intercept=False, positive=True, alpha=1 / n_samples),
                ),
            ]
        )
        self.meta_estimator_.fit(meta_features, y)
        self.meta_estimator_[1].coef_ /= self.meta_estimator_[1].coef_.sum() + 1e-9
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        X = self.scaler.transform(X)
        w = self.meta_estimator_[1].coef_[0]
        logger.info(f"w: {self.meta_estimator_[1].coef_}")
        current = [e.predict(X) for e in self.fitted_estimators_]
        prediction = w * (sum(current) / len(current))
        for i in range(len(self.noise_levels)):
            w = self.meta_estimator_[1].coef_[i + 1]
            if abs(w) < 1e-9:
                continue
            current = [
                smoothed_prediction(e, X, self.noise_quantiles * self.noise_levels[i])
                for e in self.fitted_estimators_
            ]
            prediction += w * (sum(current) / len(current))
        return prediction


def test_tree_boosting_stack():
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor

    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    estimators = [
        XGBRegressor(max_depth=1, n_estimators=5, learning_rate=1.0),
        XGBRegressor(max_depth=2, n_estimators=100, learning_rate=1.0),
        XGBRegressor(max_depth=3, n_estimators=500, learning_rate=1.0),
    ]
    model = Smooth(estimators)
    model.fit(X, y)
    pred = model.predict(X)
    print(f"Prediction shape: {pred.shape}")
    coefs = model.meta_estimator_[1].coef_
    print(f"Meta-model coefficients: {coefs}")


def test_halving_cv():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=10, random_state=42)

    # Using default param_grid
    model = GS(XGBRegressor())

    model.fit(X, y)
    pred = model.predict(X)

    print(f"Best parameters: {model.best_params_}")
    print(f"Prediction shape: {pred.shape}")


def test_horizontal():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    model = Horizontal()
    model.fit(X, y)
    pred = model.predict(X)


if __name__ == "__main__":
    test_horizontal()
