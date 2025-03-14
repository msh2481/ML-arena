import contextlib
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

import linear_models

import numpy as np
import pandas as pd
from beartype import beartype as typed
from datasets import id_map, load_dataset, split_data
from jaxtyping import Float, Int
from joblib import delayed, Parallel
from numpy import ndarray as ND
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class Arena:
    def __init__(self, n_players: int, d_outcomes: int):
        self.model = [
            LogisticRegression(fit_intercept=False) for _ in range(d_outcomes)
        ]
        self.n_players = n_players
        self.d_outcomes = d_outcomes
        self.X = np.zeros((0, 2 * n_players))
        self.y = np.zeros((0, d_outcomes))
        self.fitted = False

    @typed
    def add_data(
        self,
        first: Int[ND, "matches"],
        second: Int[ND, "matches"],
        outcome: Int[ND, "matches k"],
    ) -> None:
        n_matches = len(first)
        X = np.zeros((n_matches, 2 * self.n_players))
        y = np.zeros(n_matches)
        for i, (f, s) in enumerate(zip(first, second)):
            X[i, f] = 1
            X[i, self.n_players + s] = 1
        y = outcome
        # Add to data
        self.X = np.concatenate([self.X, X])
        self.y = np.concatenate([self.y, y])
        self.fitted = False

    @typed
    def fit(self) -> None:
        if self.fitted:
            return
        for i in range(self.d_outcomes):
            self.model[i].fit(self.X, self.y[:, i])
        self.fitted = True

    @typed
    def get_scores(self) -> Float[ND, "n_players k"]:
        full_coef = np.concatenate([model.coef_ for model in self.model], axis=0)
        assert full_coef.shape == (self.d_outcomes, 2 * self.n_players)
        first_half = full_coef[:, : self.n_players].T
        second_half = full_coef[:, self.n_players :].T
        assert np.allclose(first_half, -second_half)
        return np.round(first_half * 400)

    @typed
    def predict_win(
        self, first: Int[ND, "matches"], second: Int[ND, "matches"]
    ) -> Float[ND, "matches k"]:
        X = np.zeros((len(first), 2 * self.n_players))
        X[:, first] = 1
        X[:, self.n_players + second] = 1
        return np.stack([model.predict_proba(X)[:, 1] for model in self.model], axis=1)


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
    X_train, y_train, X_test, y_test = split_data(
        X,
        y,
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
    # Noise to break ties
    neg_mses = -np.array(mses) + np.random.randn(len(mses)) * 1e-9
    neg_maes = -np.array(maes) + np.random.randn(len(maes)) * 1e-9
    profits = np.array(profits) + np.random.randn(len(profits)) * 1e-9
    return np.stack([neg_mses, neg_maes, profits], axis=1)


@typed
def compete(
    X: Float[ND, "n_samples n_features"],
    y: Float[ND, "n_samples"],
    players: list[Callable[[ND, ND, ND], BaseEstimator]],
    matches: int,
    bad_features: bool,
    outliers: bool,
    dataset: str,
    player_names: list[str],
) -> tuple[Int[ND, "n_2"], Int[ND, "n_2"], Int[ND, "n_2 k"]]:
    metrics = Parallel(n_jobs=-1)(
        delayed(get_metrics)(X, y, players, bad_features, outliers)
        for _ in tqdm(range(matches))
    )
    metrics = np.stack(metrics, axis=0)
    df = pd.DataFrame(
        data=metrics.mean(axis=0), index=player_names, columns=["MSE", "MAE", "Profit"]
    )
    df.to_csv(
        f"results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset}.csv"
    )
    assert metrics.shape == (matches, len(players), 3)
    firsts = []
    seconds = []
    outcomes = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            current_outcomes = (
                metrics[:, np.newaxis, i, :] > metrics[np.newaxis, :, j, :]
            ).reshape(-1, 3)
            i_repeated = np.full(len(current_outcomes), i)
            j_repeated = np.full(len(current_outcomes), j)
            # i vs j
            firsts.append(i_repeated)
            seconds.append(j_repeated)
            outcomes.append(current_outcomes)
            # j vs i
            firsts.append(j_repeated)
            seconds.append(i_repeated)
            outcomes.append(1 - current_outcomes)
    firsts = np.concatenate(firsts, axis=0)
    seconds = np.concatenate(seconds, axis=0)
    outcomes = np.concatenate(outcomes, axis=0)
    return firsts, seconds, outcomes


def run_ml_arena(
    datasets: list[str],
    bad_features: bool,
    outliers: bool,
    matches: int,
):
    player_dict = {**linear_models.players}
    player_names = list(player_dict.keys())
    players = [player_dict[name] for name in player_names]
    firsts = []
    seconds = []
    outcomes = []
    for dataset in datasets:
        X, y = load_dataset(dataset)
        first, second, outcome = compete(
            X,
            y,
            players,
            matches=matches,
            bad_features=bad_features,
            outliers=outliers,
            dataset=dataset,
            player_names=player_names,
        )
        firsts.append(first)
        seconds.append(second)
        outcomes.append(outcome)
    firsts = np.concatenate(firsts, axis=0)
    seconds = np.concatenate(seconds, axis=0)
    outcomes = np.concatenate(outcomes, axis=0)
    arena = Arena(len(players), outcomes.shape[1])
    arena.add_data(firsts, seconds, outcomes)
    arena.fit()
    scores = arena.get_scores()
    score_names = ["MSE", "MAE", "Profit"]
    scores_df = pd.DataFrame(data=scores, index=player_names, columns=score_names)
    scores_df.sort_values(by="MSE", ascending=False, inplace=True)
    print(f"Datasets: {datasets}")
    print(f"Bad features: {bad_features}")
    print(f"Outliers: {outliers}")
    print(f"{matches} all-vs-all matches for each of {len(datasets)} datasets")
    print("Elo ratings for all metrics, sorted by MSE ratings:")
    print(scores_df)
    return scores_df


def test_new_arena():
    # Simple test for the new arena implementation
    winners = np.array([0, 1, 0, 1, 2, 2])
    losers = np.array([1, 2, 2, 0, 1, 0])
    outcomes = np.array([1, 1, 1, 0, 0, 0])
    outcomes = np.stack(
        [
            outcomes,
            1 - outcomes,
        ],
        axis=1,
    )
    print(outcomes.shape)
    arena = Arena(n_players=3, d_outcomes=2)
    arena.add_data(winners, losers, outcomes)
    arena.fit()

    scores = np.round(arena.get_scores(), 2)
    print("Scores after training:")
    print(scores)

    def w(i, j):
        return arena.predict_win(np.array([i]), np.array([j])).squeeze()

    print("\nWin probabilities:")
    print(f"P(A wins over B): {w(0, 1)}")
    print(f"P(B wins over C): {w(1, 2)}")
    print(f"P(A wins over C): {w(0, 2)}")


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
    df = run_ml_arena(
        datasets=datasets,
        bad_features=True,
        outliers=True,
        matches=10,
    )
    filename = f"results/arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename)
