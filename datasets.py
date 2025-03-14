import numpy as np
import pandas as pd
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
from sklearn.linear_model import (
    ARDRegression,
    Lasso,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

id_map = {
    "student_performance": 320,
    "concrete": 165,
    "computer_hardware": 29,
    "kidney_disease": 857,
    "fertility": 244,
    "algerian_forest_fires": 547,
    "airfoil_self_noise": 291,
    "istanbul_stock_exchange": 247,
}

default_target_cols = {
    "computer_hardware": "ERP",
    "istanbul_stock_exchange": "ISE",
}


@typed
def load_dataset(
    name: str,
) -> tuple[Float[ND, "n_samples n_features"], Float[ND, "n_samples ..."]]:
    data = fetch_ucirepo(id=id_map[name])
    X = data.data.features
    # Drop duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    y = data.data.targets
    if isinstance(y, pd.DataFrame):
        y = y[y.columns[0]]
    metadata = data.metadata
    variables = data.variables
    if y is None and name in default_target_cols:
        target_col = default_target_cols[name]
        y = X[target_col].copy()
        X = X.drop(columns=[target_col])
    # Handle categorical features by encoding them
    X_processed = X.copy()
    for column in X_processed.columns:
        if X_processed[column].dtype == "object":
            le = LabelEncoder()
            X_processed.loc[:, column] = le.fit_transform(X_processed[column])
    # Remove features with only one unique value
    columns_to_keep = []
    for column in X_processed.columns:
        if len(X_processed[column].unique()) >= 2:
            columns_to_keep.append(column)

    X_processed = X_processed[columns_to_keep]
    X = X_processed.values.astype(np.float32)
    assert isinstance(y, pd.Series)
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
    if isinstance(y, pd.Series):
        y = y.values
    y = y.astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return X, y


def test_load_dataset():
    for dataset_name in id_map.keys():
        try:
            X, y = load_dataset(dataset_name)
            assert isinstance(X, np.ndarray), f"{dataset_name}: X is not a numpy array"
            assert isinstance(y, np.ndarray), f"{dataset_name}: y is not a numpy array"
            assert np.issubdtype(
                X.dtype, np.floating
            ), f"{dataset_name}: X does not contain float values"
            assert np.issubdtype(
                y.dtype, np.floating
            ), f"{dataset_name}: y does not contain float values"
            assert X.ndim == 2, f"{dataset_name}: X is not 2-dimensional"
            assert y.ndim == 1, f"{dataset_name}: y is not 1-dimensional"
            n_samples = X.shape[0]
            assert (
                y.shape[0] == n_samples
            ), f"{dataset_name}: X and y have different number of samples"
            assert not np.any(np.isnan(X)), f"{dataset_name}: X contains NaN values"
            assert not np.any(np.isnan(y)), f"{dataset_name}: y contains NaN values"
            assert not np.any(
                np.isinf(X)
            ), f"{dataset_name}: X contains infinite values"
            assert not np.any(
                np.isinf(y)
            ), f"{dataset_name}: y contains infinite values"
            assert np.all(X >= -1e9) and np.all(
                X <= 1e9
            ), f"{dataset_name}: X contains values outside of [-1e9, 1e9]"
            assert np.all(y >= -1e9) and np.all(
                y <= 1e9
            ), f"{dataset_name}: y contains values outside of [-1e9, 1e9]"
            for i in range(X.shape[1]):
                assert (
                    len(np.unique(X[:, i])) >= 2
                ), f"{dataset_name}: Feature {i} has less than 2 unique values"
            assert (
                len(np.unique(y)) >= 2
            ), f"{dataset_name}: Target has less than 2 unique values"

            print(
                f"Dataset {dataset_name} loaded successfully: X shape {X.shape}, y shape {y.shape}"
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue  # Continue to the next dataset instead of returning


@typed
def split_data(
    X: Float[ND, "n_samples n_features"],
    y: Float[ND, "n_samples"],
    test_size: float = 0.25,
    outliers: bool = False,
    bad_features: bool = False,
) -> tuple[
    Float[ND, "n_samples n_features"],
    Float[ND, "n_samples"],
    Float[ND, "n_test n_features"],
    Float[ND, "n_test"],
]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    random_subset = lambda n, k: np.random.choice(n, k, replace=False)
    shuffle = lambda x: x[np.random.permutation(len(x))]
    n_samples, n_features = X_train.shape
    if outliers:
        outlier_indices = random_subset(n_samples, n_samples // 4)
        y_train[outlier_indices] = shuffle(y_train[outlier_indices])
    if bad_features:
        bad_feature_indices = random_subset(n_features, n_features // 2)
        X_train[:, bad_feature_indices] = shuffle(X_train[:, bad_feature_indices])
    return X_train, y_train, X_test, y_test


def test_split_data():
    X, y = load_dataset("computer_hardware")
    X_train, y_train, X_test, y_test = split_data(
        X,
        y,
        outliers=False,
        bad_features=False,
    )
    correlations = []
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)
    correlations = (100 * correlations).astype(int)
    print(correlations)
    model = Lasso(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ard_coefs = model.coef_
    print(np.square(pred - y_test).mean())
    model = LassoLarsCV()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ridge_coefs = model.coef_
    print(np.square(pred - y_test).mean())
    print(ard_coefs)
    print(ridge_coefs)


def save_all_datasets():
    for dataset_name in id_map.keys():
        X, y = load_dataset(dataset_name)
        np.save(f"data/{dataset_name}_X.npy", X)
        np.save(f"data/{dataset_name}_y.npy", y)


if __name__ == "__main__":
    # test_split_data()
    save_all_datasets()
