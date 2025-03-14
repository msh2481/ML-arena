import numpy as np
import pandas as pd
from beartype import beartype as typed
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND
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

# Default target columns for datasets where target_col is None
default_target_cols = {
    "computer_hardware": "ERP",
    "istanbul_stock_exchange": "ISE",
}


@typed
def load_dataset(
    name: str,
) -> tuple[Float[ND, "n_samples n_features"], Float[ND, "n_samples ..."]]:
    logger.info(f"Loading dataset {name}")
    data = fetch_ucirepo(id=id_map[name])
    X = data.data.features
    # Drop duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    y = data.data.targets
    if isinstance(y, pd.DataFrame):
        y = y[y.columns[0]]
    metadata = data.metadata
    variables = data.variables
    # logger.info(f"Metadata: {metadata}")
    # logger.info(f"Variables: {variables}")

    if y is None and name in default_target_cols:
        target_col = default_target_cols[name]
        y = X[target_col].copy()
        X = X.drop(columns=[target_col])

    logger.info(f"X: {type(X)} {X.shape} {X.columns} | y: {type(y)} {y.shape}")
    # Handle categorical features by encoding them
    X_processed = X.copy()
    for column in X_processed.columns:
        if X_processed[column].dtype == "object":
            le = LabelEncoder()
            X_processed.loc[:, column] = le.fit_transform(X_processed[column])

    # Convert pandas DataFrames to NumPy arrays
    X = X_processed.values.astype(np.float32)
    assert isinstance(y, pd.Series)

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
    if isinstance(y, pd.Series):
        y = y.values
    y = y.astype(np.float32)
    return X, y


def test_load_dataset():
    for dataset_name in id_map.keys():
        try:
            X, y = load_dataset(dataset_name)

            # Check that X and y are numpy arrays
            assert isinstance(X, np.ndarray), f"{dataset_name}: X is not a numpy array"
            assert isinstance(y, np.ndarray), f"{dataset_name}: y is not a numpy array"

            # Check that X and y contain float values
            assert np.issubdtype(
                X.dtype, np.floating
            ), f"{dataset_name}: X does not contain float values"
            assert np.issubdtype(
                y.dtype, np.floating
            ), f"{dataset_name}: y does not contain float values"

            # Check shapes
            assert X.ndim == 2, f"{dataset_name}: X is not 2-dimensional"
            assert y.ndim <= 2, f"{dataset_name}: y has more than 2 dimensions"

            # Check that X and y have the same number of samples
            n_samples = X.shape[0]
            if y.ndim == 1:
                assert (
                    y.shape[0] == n_samples
                ), f"{dataset_name}: X and y have different number of samples"
            else:
                assert (
                    y.shape[0] == n_samples
                ), f"{dataset_name}: X and y have different number of samples"

            print(
                f"Dataset {dataset_name} loaded successfully: X shape {X.shape}, y shape {y.shape}"
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue  # Continue to the next dataset instead of returning


if __name__ == "__main__":
    test_load_dataset()
