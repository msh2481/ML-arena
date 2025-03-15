from typing import Any

import numpy as np
import torch
from beartype import beartype as typed
from beartype.typing import Literal
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
from scipy.stats import norm as gaussian, t as student_t
from sklearn.base import BaseEstimator, clone, RegressorMixin
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.utils import check_random_state


class Iso(BaseEstimator, RegressorMixin):
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.isotonic.fit(self.estimator.predict(X), y)
        return self

    def predict(self, X):
        return self.isotonic.predict(self.estimator.predict(X))


class BFS(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: Any,
        cv: int = 5,
        use_scaling: bool = True,
        use_positive: bool = True,
        use_intercept: bool = False,
        add_singles: bool = False,
        final: Literal["ridge", "lasso", "lasso_cv", "ard"] = "ard",
    ):
        self.estimator = estimator
        self.estimators_ = []
        self.feature_indices_ = []
        self.cv = cv
        self.use_scaling = use_scaling
        self.use_positive = use_positive
        self.use_intercept = use_intercept
        self.add_singles = add_singles
        self.final = final

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        n_samples, n_features = X.shape
        # Fit estimator for each feature separately and compute its cross-validated MSE
        meta_features_single = np.zeros((n_samples, n_features))
        errors_single = np.zeros((n_features, n_samples))
        for i in range(n_features):
            kf = KFold(n_splits=self.cv, shuffle=False)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx][:, [i]], X[test_idx][:, [i]]
                assert X_train.shape == (len(train_idx), 1)
                assert X_test.shape == (len(test_idx), 1)
                y_train, y_test = y[train_idx], y[test_idx]
                estimator = clone(self.estimator)
                estimator.fit(X_train, y_train)
                pred = estimator.predict(X_test)
                meta_features_single[test_idx, i] = pred
                errors_single[i, test_idx] = y_test - pred
        mse = np.mean(errors_single**2, axis=1)
        # Sort features by increasing MSE and train estimator on every prefix
        sorted_indices = np.argsort(mse)
        for i in range(1, n_features + 1):
            estimator = clone(self.estimator)
            estimator.fit(X[:, sorted_indices[:i]], y)
            self.estimators_.append(estimator)
            self.feature_indices_.append(sorted_indices[:i])
        # Train meta-estimator on their cross-validated predictions
        meta_features = np.zeros((n_samples, n_features))
        for i, feats in enumerate(self.feature_indices_):
            kf = KFold(n_splits=self.cv, shuffle=False)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx][:, feats], X[test_idx][:, feats]
                y_train, y_test = y[train_idx], y[test_idx]
                assert X_train.shape == (len(train_idx), len(feats))
                assert X_test.shape == (len(test_idx), len(feats))
                assert y_train.shape == (len(train_idx),)
                assert y_test.shape == (len(test_idx),)
                estimator = clone(self.estimator)
                estimator.fit(X_train, y_train)
                meta_features[test_idx, i] = estimator.predict(X_test)
        if self.add_singles:
            meta_features = np.concatenate(
                [meta_features, meta_features_single], axis=1
            )
            for i in range(n_features):
                estimator = clone(self.estimator)
                estimator.fit(X[:, [i]], y)
                self.estimators_.append(estimator)
                self.feature_indices_.append([i])

        if self.final == "ridge":
            inner = Ridge(
                alpha=1 / (2 * len(meta_features)),
                positive=self.use_positive,
                fit_intercept=self.use_intercept,
            )
        elif self.final == "lasso":
            inner = Lasso(
                alpha=1 / (2 * len(meta_features)),
                positive=self.use_positive,
                fit_intercept=self.use_intercept,
            )
        elif self.final == "lasso_cv":
            inner = LassoCV(
                cv=self.cv, positive=self.use_positive, fit_intercept=self.use_intercept
            )
        elif self.final == "ard":
            inner = ARDRegression(fit_intercept=self.use_intercept)
        else:
            raise ValueError(f"Invalid final model: {self.final}")
        if self.use_scaling:
            self.meta_estimator_ = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "estimator",
                        inner,
                    ),
                ]
            )
        else:
            self.meta_estimator_ = inner
        self.meta_estimator_.fit(meta_features, y)
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        meta_features = np.zeros((X.shape[0], len(self.estimators_)))
        for i, (estimator, feature_indices) in enumerate(
            zip(self.estimators_, self.feature_indices_)
        ):
            meta_features[:, i] = estimator.predict(X[:, feature_indices])
        return self.meta_estimator_.predict(meta_features)


class RFS(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: Any,
        n_estimators: int = 10,
        max_features: float | int = 0.9,
        random_state: int | None = None,
        use_scaling: bool = True,
        use_intercept: bool = False,
        use_positive: bool = True,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.use_scaling = use_scaling
        self.use_positive = use_positive
        self.use_intercept = use_intercept

    @typed
    def fit(
        self, X: Float[ND, "n d_in"] | torch.Tensor, y: Float[ND, "n"] | torch.Tensor
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        n_samples, n_features = X.shape
        if isinstance(self.max_features, float):
            self.max_features = int(np.ceil(self.max_features * n_features))
        # chance to cover all feature = (1 - (1 - r)^n_estimators)^n_features
        # where r = max_features / n_features
        # set 1 - (1 - r)^n_estimators = 0.95^(1/n_features)
        # r  = 1 - (1 - 0.95^(1/n_features))^(1/n_estimators)
        prob_for_one = 0.95 ** (1 / n_features)
        lower_bound_ratio = 1 - ((1 - prob_for_one) ** (1 / self.n_estimators))
        lower_bound = int(np.ceil(n_features * lower_bound_ratio))
        if self.max_features < lower_bound:
            logger.warning(
                f"Max features {self.max_features} -> {lower_bound} to cover all features"
            )
            self.max_features = lower_bound
        random_state = check_random_state(self.random_state)
        kf = KFold(n_splits=2, shuffle=True, random_state=random_state)
        self.estimators_ = []
        self.feature_indices_ = []
        for i in range(self.n_estimators):
            # Sample features until we get a new subset
            feature_indices = None
            max_attempts = 10  # Avoid infinite loop
            for _ in range(max_attempts):
                candidate_indices = random_state.choice(
                    n_features, size=self.max_features, replace=False
                )
                candidate_indices = np.sort(candidate_indices)
                if feature_indices is None or all(
                    not np.array_equal(candidate_indices, prev_indices)
                    for prev_indices in self.feature_indices_
                ):
                    feature_indices = candidate_indices
                    break
            # If we couldn't find a new subset after max attempts, just use the last one
            if feature_indices is None:
                feature_indices = candidate_indices
            self.feature_indices_.append(feature_indices)
            estimator = clone(self.estimator)
            estimator.fit(X[:, feature_indices], y)
            self.estimators_.append(estimator)

        meta_features = np.zeros((n_samples, self.n_estimators))
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            for j in range(self.n_estimators):
                feature_indices = self.feature_indices_[j]
                temp_estimator = clone(self.estimator)
                temp_estimator.fit(X_train[:, feature_indices], y_train)
                meta_features[test_idx, j] = temp_estimator.predict(
                    X_test[:, feature_indices]
                )
        if self.use_scaling:
            self.meta_estimator_ = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "estimator",
                        Ridge(
                            alpha=1 / (2 * len(meta_features)),
                            positive=self.use_positive,
                            fit_intercept=self.use_intercept,
                        ),
                    ),
                ]
            )
        else:
            self.meta_estimator_ = Ridge(
                alpha=1 / (2 * len(meta_features)),
                positive=self.use_positive,
                fit_intercept=self.use_intercept,
            )
        self.meta_estimator_.fit(meta_features, y)
        return self

    @typed
    def predict(self, X: Float[ND, "m d_in"] | torch.Tensor) -> Float[ND, "m"]:
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()

        meta_features = np.zeros((X.shape[0], self.n_estimators))
        for i, (estimator, feature_indices) in enumerate(
            zip(self.estimators_, self.feature_indices_)
        ):
            meta_features[:, i] = estimator.predict(X[:, feature_indices])
        return self.meta_estimator_.predict(meta_features)


class RobustRegressor(BaseEstimator, RegressorMixin):
    @typed
    def __init__(
        self,
        estimator: Any,
        iterations: int = 5,
    ):
        self.estimator = estimator
        self.iterations = iterations

    @typed
    def _compute_sample_probs(self) -> np.ndarray:
        iters = self.sample_errors.shape[1]
        sample_mse = np.mean(self.sample_errors**2, axis=1)
        mean_error = np.median(sample_mse) + 1e-8
        weights = 1.0 / (iters * sample_mse + 2 * mean_error)
        weights /= np.max(weights)
        return weights

    @typed
    def _bootstrap_sample(
        self, indices: Int[ND, "n_samples"], k: int = 6
    ) -> Int[ND, "n_samples"]:
        if self.sample_probs is None:
            return indices
        selected = self.sample_probs[indices]
        counts = (k * selected / selected.max()).astype(np.int32)
        # Take indices[i] counts[i] times
        return np.repeat(indices, counts)

    @typed
    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()

        self.n_samples, self.n_features = X.shape
        self.sample_errors = np.zeros((self.n_samples, 0))
        self.sample_probs = None
        self.estimators = []

        for i_it in range(self.iterations):
            kf = KFold(n_splits=2, shuffle=True, random_state=i_it)
            current_sample_errors = np.zeros((self.n_samples))
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                bootstrap_indices = self._bootstrap_sample(train_idx)
                X_train, X_test = X[bootstrap_indices], X[test_idx]
                y_train, y_test = y[bootstrap_indices], y[test_idx]
                current_estimator = clone(self.estimator)
                current_estimator.fit(X_train, y_train)
                y_pred = current_estimator.predict(X_test)
                current_sample_errors[test_idx] = y_test - y_pred
            self.sample_errors = np.concatenate(
                [self.sample_errors, current_sample_errors[:, None]], axis=1
            )
            # Remove 20% of the oldest sample errors
            if (i_it + 1) % 5 == 0:
                self.sample_errors = self.sample_errors[:, 1:]
            self.sample_probs = self._compute_sample_probs()
            self.sample_probs = self.sample_probs
            full_idx = self._bootstrap_sample(np.arange(self.n_samples))
            current_estimator = clone(self.estimator)
            current_estimator.fit(X[full_idx], y[full_idx])
            self.estimators.append(current_estimator)
        return self

    @typed
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict using the ensemble of estimators."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        assert self.estimators, "Model not fitted yet or no estimators in ensemble"
        predictions = []
        for i, estimator in enumerate(self.estimators):
            pred = estimator.predict(X)
            predictions.append(pred)
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)


def wrap(
    model: BaseEstimator,
    scale: Literal["standard", "robust", "quantile", "id"] = "standard",
    feats: Literal["bfs", "rfs", "id"] = "id",
    outliers: Literal["robust", "id"] = "id",
    isotonic: bool = False,
) -> BaseEstimator:
    scaled_model = None
    if scale == "standard":
        scaled_model = Pipeline([("scaler", StandardScaler()), ("model", model)])
    elif scale == "robust":
        scaled_model = Pipeline([("scaler", RobustScaler()), ("model", model)])
    elif scale == "quantile":
        scaled_model = Pipeline([("scaler", QuantileTransformer()), ("model", model)])
    elif scale == "id":
        scaled_model = model
    else:
        raise ValueError(f"Invalid scaling method: {scale}")
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


def test_backward_feature_stacking_regressor():
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model = BFS(Ridge(), add_singles=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (100,)


if __name__ == "__main__":
    test_backward_feature_stacking_regressor()
