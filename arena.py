import contextlib
import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from beartype import beartype as typed
from datasets import load_dataset, split_data
from jaxtyping import Float
from kernel_models import KernelKNN
from lightgbm import LGBMRegressor
from linear_models import MISO, wrap
from loguru import logger
from numpy import ndarray as ND
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression, LinearRegression, Ridge
from tqdm.auto import tqdm
from tree_models import GS, Horizontal, Hybrid, XGBRegressor
from sklearn.model_selection import GridSearchCV as GS


class Silencer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.null_file = None
        self.stdout_redirector = None
        self.stderr_redirector = None

    def __enter__(self):
        if not self.verbose:
            self.null_file = open(os.devnull, "w")
            self.stdout_redirector = contextlib.redirect_stdout(self.null_file)
            self.stderr_redirector = contextlib.redirect_stderr(self.null_file)
            self.stdout_redirector.__enter__()
            self.stderr_redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            self.stderr_redirector.__exit__(exc_type, exc_val, exc_tb)
            self.stdout_redirector.__exit__(exc_type, exc_val, exc_tb)
            self.null_file.close()


@typed
def get_metrics(
    X: Float[ND, "n_samples n_features"],
    y: Float[ND, "n_samples"],
    players: list[Callable[[ND, ND, ND], BaseEstimator]],
    bad_features: bool,
    outliers: bool,
) -> Float[ND, "n_players 3"]:
    n_samples = X.shape[0]
    X_train, y_train, X_test, y_test = split_data(
        X,
        y,
        test_size=0.5,
        bad_features=bad_features,
        outliers=outliers,
    )
    mses = []
    maes = []
    profits = []
    # Metrics
    for player_fn in players:
        player = player_fn(X_train, y_train)
        with Silencer():
            player.fit(X_train, y_train)
            p = player.predict(X_test)
        current_mse = np.square(p - y_test).mean()
        current_mae = np.abs(p - y_test).mean()
        current_profit = (y_test * (p > 0)).mean()
        mses.append(current_mse)
        maes.append(current_mae)
        profits.append(current_profit)
    return np.stack([np.array(mses), np.array(maes), np.array(profits)], axis=1)


@typed
def collect_metrics(
    X: Float[ND, "n_samples n_features"],
    y: Float[ND, "n_samples"],
    players: list[Callable[[ND, ND, ND], BaseEstimator]],
    matches: int,
    bad_features: bool,
    outliers: bool,
    dataset: str,
    player_names: list[str],
) -> Float[ND, "n_players 3"]:
    logger.info(f"Running {dataset}...")
    metrics_list = [
        get_metrics(X, y, players, bad_features, outliers) for _ in tqdm(range(matches))
    ]
    metrics = np.stack(metrics_list, axis=0)
    avg_metrics = metrics.mean(axis=0)
    df = pd.DataFrame(
        data=avg_metrics, index=player_names, columns=["MSE", "MAE", "Profit"]
    )
    df.to_csv(
        f"results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset}.csv"
    )
    return avg_metrics


def run_ml_evaluation(
    datasets: list[str],
    bad_features: bool,
    outliers: bool,
    matches: int,
):
    player_dict = {
        # MISO
        "MISO": lambda x, _: MISO(),
        # Hybrid models
        "Hybrid": lambda x, _: Hybrid(tree_type="lgbm"),
        "Horizontal": lambda x, _: Horizontal(),
        # "GS+Hybrid": lambda x, _: Hybrid(use_gs=True, tree_type="lgbm"),
        # LGBM
        # "GS+LGBM": lambda x, _: GS(LGBMRegressor(learning_rate=0.05, max_depth=3)),
        "LGBM": lambda x, _: LGBMRegressor(learning_rate=0.05, max_depth=3),
        # Baselines
        "RandomForest": lambda x, _: RandomForestRegressor(
            n_estimators=100, max_depth=3
        ),
        "IsoBFS(ARDRegression)": lambda x, _: wrap(
            ARDRegression(), feats="bfs", isotonic=True
        ),
        "Ridge": lambda x, _: wrap(Ridge(1 / (2 * len(x)))),
        "LinearRegression": lambda x, _: wrap(LinearRegression()),
    }
    player_names = list(player_dict.keys())
    players = [player_dict[name] for name in player_names]
    all_metrics = []

    for dataset in datasets:
        X, y = load_dataset(dataset)
        avg_metrics = collect_metrics(
            X,
            y,
            players,
            matches=matches,
            bad_features=bad_features,
            outliers=outliers,
            dataset=dataset,
            player_names=player_names,
        )
        all_metrics.append(avg_metrics)

    # Average metrics across all datasets
    all_metrics_array = np.stack(all_metrics, axis=0)
    overall_metrics = all_metrics_array.mean(axis=0)

    # Create DataFrame with overall metrics
    score_names = ["MSE", "MAE", "Profit"]
    metrics_df = pd.DataFrame(
        data=overall_metrics, index=player_names, columns=score_names
    )

    # Sort by MSE (lower is better)
    metrics_df["Score"] = metrics_df["Profit"] / (
        (metrics_df["MSE"] * metrics_df["MAE"]) ** 0.5
    )
    metrics_df.sort_values(by="Score", ascending=False, inplace=True)

    print(f"Datasets: {datasets}")
    print(f"Bad features: {bad_features}")
    print(f"Outliers: {outliers}")
    print(f"{matches} evaluations for each of {len(datasets)} datasets")
    print("Average metrics across all datasets, sorted by MSE:")
    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    datasets = [
        "student_performance",
        "concrete",
        "computer_hardware",
        "kidney_disease",
        "fertility",
        "algerian_forest_fires",
        "airfoil_self_noise",
        "istanbul_stock_exchange",
    ]
    df = run_ml_evaluation(
        datasets=datasets,
        bad_features=False,
        outliers=True,
        matches=10,
    )
    mse = df["MSE"]
    mae = df["MAE"]
    profit = df["Profit"]
    # Compute correlation of mse and mae with profit
    corr_mse = mse.corr(profit)
    corr_mae = mae.corr(profit)
    print(f"corr(mse, profit) = {corr_mse:.2f}")
    print(f"corr(mae, profit) = {corr_mae:.2f}")
    filename = f"results/metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename)
