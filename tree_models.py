import numpy as np
import torch
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from lightgbm import LGBMRegressor
from loguru import logger
from numpy import ndarray as ND
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.experimental import enable_halving_search_cv
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
from sklearn.model_selection import GridSearchCV as GS, HalvingGridSearchCV, KFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from wrappers import BFS, Iso, RFS, RobustRegressor, wrap
from xgboost import XGBRegressor


class Hybrid(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        use_hcv: bool = False,
        use_isotonic: bool = False,
        scale: Literal["id", "standard", "robust"] = "standard",
        feats: Literal["id", "bfs", "rfs"] = "bfs",
    ):
        self.use_hcv = use_hcv
        self.use_isotonic = use_isotonic
        self.feats = feats
        self.linear = wrap(
            ARDRegression(),
            scale=scale,
            feats=feats,
            isotonic=False,
        )
        self.tree = XGBRegressor(learning_rate=0.05, max_depth=3)
        if self.use_hcv:
            self.tree = HCV(self.tree)
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

        self.linear.fit(X, y)
        linear_pred = self.linear.predict(X)
        self.tree.fit(X, y - linear_pred)
        tree_pred = self.tree.predict(X)
        if self.use_isotonic:
            self.isotonic.fit(linear_pred + tree_pred, y)
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        linear_pred = self.linear.predict(X)
        tree_pred = self.tree.predict(X)
        if self.use_isotonic:
            isotonic_pred = self.isotonic.predict(linear_pred + tree_pred)
            return isotonic_pred
        return linear_pred + tree_pred


class HCV(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: BaseEstimator,
        top_k: int = 3,
        resource: str = "n_samples",
        min_resources: int | str = 50,
        max_resources: int | str = "auto",
        max_depths: list[int] = [2, 3],
        learning_rates: list[float] = [0.02, 0.05, 0.1, 0.2],
        aggressive_elimination: bool = False,
        cv: int = 2,
        random_state: int = 42,
        factor: int = 3,
        run_full: bool = False,
    ):
        self.estimator = estimator
        self.top_k = top_k
        if run_full:
            max_depths = [2, 3, 4]
            learning_rates = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        self.param_grid = {
            "max_depth": max_depths,
            "learning_rate": learning_rates,
        }
        self.factor = factor
        self.resource = resource
        self.min_resources = min_resources
        self.max_resources = max_resources
        self.aggressive_elimination = aggressive_elimination
        self.cv = cv
        self.random_state = random_state

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        self.search_ = HalvingGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            factor=self.factor,
            resource=self.resource,
            min_resources=self.min_resources,
            max_resources=self.max_resources,
            aggressive_elimination=self.aggressive_elimination,
            cv=self.cv,
            random_state=self.random_state,
        )

        self.search_.fit(X, y)
        self.best_params_ = self.search_.best_params_

        # Get top-3 models from the search results
        results = self.search_.cv_results_
        indices = np.argsort(results["mean_test_score"])[-self.top_k :]

        estimators = []
        for idx in indices:
            params = {k: results["param_" + k][idx] for k in self.param_grid.keys()}
            logger.info(f"Best parameters #{idx}: {params}")
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
    model = HCV(
        estimator=XGBRegressor(),
        resource="n_estimators",
        factor=2,
        cv=3,
        random_state=42,
    )

    model.fit(X, y)
    pred = model.predict(X)

    print(f"Best parameters: {model.best_params_}")
    print(f"Prediction shape: {pred.shape}")


if __name__ == "__main__":
    test_halving_cv()

# players = {
#     "DecisionTreeRegressor": lambda x, _: wrap(
#         GS(DecisionTreeRegressor(), param_grid={"max_depth": [2, 4, 6]}), scale="id"
#     ),
#     "RandomForestRegressor": lambda x, _: wrap(
#         GS(RandomForestRegressor(), param_grid={"max_depth": [2, 4, 6]}), scale="id"
#     ),
#     "GradientBoostingRegressor": lambda x, _: wrap(
#         GS(
#             GradientBoostingRegressor(),
#             param_grid={"max_depth": [2, 4, 6]},
#         ),
#         scale="id",
#     ),
#     "XGBRegressor": lambda x, _: wrap(
#         GS(XGBRegressor(), param_grid={"max_depth": [2, 4, 6]}), scale="id"
#     ),
#     "LGBMRegressor": lambda x, _: wrap(
#         GS(LGBMRegressor(), param_grid={"max_depth": [2, 4, 6]}), scale="id"
#     ),
#     "XGBRegressor_Halving": lambda x, _: wrap(
#         HalvingCV(
#             XGBRegressor(),
#             resource="n_estimators",
#         ),
#         scale="id",
#     ),
#     "LGBMRegressor_Halving": lambda x, _: wrap(
#         HalvingCV(
#             LGBMRegressor(),
#             resource="n_estimators",
#         ),
#         scale="id",
#     ),
#     "GradientBoostingRegressor_Halving": lambda x, _: wrap(
#         HalvingCV(
#             GradientBoostingRegressor(),
#             resource="n_estimators",
#         ),
#         scale="id",
#     ),
# }
