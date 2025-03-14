from collections import defaultdict
from typing import Any, Callable

import linear_models

import numpy as np
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
        full_coef = np.concatenate([model.coef_ for model in self.model], axis=1)
        first_half = full_coef[: self.n_players]
        second_half = full_coef[self.n_players :]
        assert np.allclose(first_half, -second_half)
        return first_half

    @typed
    def predict_win(
        self, first: Int[ND, "matches"], second: Int[ND, "matches"]
    ) -> Float[ND, "matches k"]:
        X = np.zeros((len(first), 2 * self.n_players))
        X[:, first] = 1
        X[:, self.n_players + second] = 1
        return np.stack([model.predict_proba(X)[:, 1] for model in self.model], axis=1)


@typed
def compete(
    X: Float[ND, "n_samples n_features"],
    y: Float[ND, "n_samples"],
    players: list[Callable[[ND, ND, ND], BaseEstimator]],
) -> Int[ND, "n_2 k"]:
    X_train, y_train, X_test, y_test = split_data(X, y)
    mses = []
    maes = []
    profits = []
    # Metrics
    for player_fn in players:
        player = player_fn(X_train, y_train)
        player.fit(X_train, y_train)
        p = player.predict(X_test)
        mse = np.square(p - y_test).mean()
        mae = np.abs(p - y_test).mean()
        profit = (y_test * (p > 0)).mean()
        mses.append(mse)
        maes.append(mae)
        profits.append(profit)
    mses = np.array(mses)
    maes = np.array(maes)
    profits = np.array(profits)
    # Outcomes
    first, second = np.mgrid[: len(players), : len(players)]
    first = first.flatten()
    second = second.flatten()
    return np.stack(
        [
            mse[first] < mse[second],
            mae[first] < mae[second],
            profits[first] > profits[second],
        ],
        axis=1,
    ).astype(int)


def run_ml_arena(n_matches: int = 10**4, n_bootstrap: int = 10**3):
    player_dict = {**linear_models.players}
    player_names = list(player_dict.keys())
    players = [player_dict[name] for name in player_names]
    X, y = load_dataset("computer_hardware")
    compete(X, y, players)


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
    # run_ml_arena()
    test_new_arena()
