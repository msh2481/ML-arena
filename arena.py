from collections import defaultdict
from typing import Any, Callable

import numpy as np
from beartype import beartype as typed
from jaxtyping import Float, Int
from joblib import delayed, Parallel
from numpy import ndarray as ND
from openskill.models import BradleyTerryFull
from sklearn.linear_model import (
    ARDRegression,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm


class Arena:
    def __init__(self, n_players: int):
        self.model = LogisticRegression(fit_intercept=False)
        self.n_players = n_players
        self.X = np.zeros((0, 2 * n_players))
        self.y = np.zeros(0)
        self.fitted = False

    @typed
    def add_data(self, winners: Int[ND, "matches"], losers: Int[ND, "matches"]) -> None:
        n_matches = len(winners)
        X = np.zeros((2 * n_matches, 2 * self.n_players))
        y = np.zeros(2 * n_matches)

        # For each match, add two training examples:
        # (winner, loser, 1) and (loser, winner, 0)
        for i in range(n_matches):
            # Winner vs loser (win)
            X[2 * i, winners[i]] = 1
            X[2 * i, self.n_players + losers[i]] = 1
            y[2 * i] = 1

            # Loser vs winner (loss)
            X[2 * i + 1, losers[i]] = 1
            X[2 * i + 1, self.n_players + winners[i]] = 1
            y[2 * i + 1] = 0

        self.X = np.concatenate([self.X, X])
        self.y = np.concatenate([self.y, y])
        self.fitted = False

    @typed
    def fit(self) -> None:
        if self.fitted:
            return
        self.model.fit(self.X, self.y)
        self.fitted = True

    @typed
    def get_scores(self) -> Float[ND, "n_players"]:
        return self.model.coef_[0, : self.n_players]

    @typed
    def predict_win(
        self, first: Int[ND, "matches"], second: Int[ND, "matches"]
    ) -> Float[ND, "matches"]:
        X = np.zeros((len(first), 2 * self.n_players))
        X[:, first] = 1
        X[:, self.n_players + second] = 1
        return self.model.predict_proba(X)[:, 1]


@typed
def generate_dataset(
    outliers: bool = False,
    bad_features: bool = False,
    n_samples: int = 1000,
    n_features: int = 20,
) -> tuple[
    Float[ND, "n_samples n_features"],
    Float[ND, "n_samples"],
    Float[ND, "n_test n_features"],
    Float[ND, "n_test"],
]:
    """Generate a dataset based on a linear model with noise.

    Args:
        outliers: If True, shuffle 25% of Y_train values
        bad_features: If True, shuffle half of the features
        n_samples: Number of training samples
        n_features: Number of features

    Returns:
        X_train, Y_train, X_test, Y_test
    """
    # Generate true coefficients
    true_coef = np.random.randn(n_features)

    # Generate training data
    X_train = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) * 0.1
    Y_train = X_train @ true_coef + noise

    # Generate test data
    n_test = n_samples // 5
    X_test = np.random.randn(n_test, n_features)
    noise_test = np.random.randn(n_test) * 0.1
    Y_test = X_test @ true_coef + noise_test

    # Add outliers if requested
    if outliers:
        outlier_indices = np.random.choice(
            n_samples, size=n_samples // 4, replace=False
        )
        Y_train_outliers = Y_train[outlier_indices].copy()
        Y_train[outlier_indices] = Y_train_outliers[
            np.random.permutation(len(Y_train_outliers))
        ]

    # Add bad features if requested
    if bad_features:
        bad_feature_indices = np.random.choice(
            n_features, size=n_features // 2, replace=False
        )
        for feature_idx in bad_feature_indices:
            X_train_feature = X_train[:, feature_idx].copy()
            X_train[:, feature_idx] = X_train_feature[
                np.random.permutation(len(X_train))
            ]

    return X_train, Y_train, X_test, Y_test


def ridge_adaptive(X_train, y_train, X_test):
    alpha = 1.0 / (2 * len(X_train))
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ridge_fixed(X_train, y_train, X_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ard(X_train, y_train, X_test):
    model = ARDRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ridgecv(X_train, y_train, X_test):
    model = RidgeCV(alphas=10 ** np.linspace(-4, 2, 20), cv=3)
    model.fit(X_train, y_train)
    return model.predict(X_test)


players = {
    "Ridge(adaptive)": ridge_adaptive,
    "Ridge(1.0)": ridge_fixed,
    "ARDRegression": ard,
    "LinearRegression": linear,
    "RidgeCV": ridgecv,
}


@typed
def compete(player1: str, player2: str) -> bool:
    X_train, y_train, X_test, y_test = generate_dataset(
        outliers=False, bad_features=True
    )
    pred1 = players[player1](X_train, y_train, X_test)
    pred2 = players[player2](X_train, y_train, X_test)
    mse1 = mean_squared_error(y_test, pred1)
    mse2 = mean_squared_error(y_test, pred2)
    return mse1 < mse2


def run_ml_arena(n_matches: int = 10**4, n_bootstrap: int = 10**3):
    player_names = list(players.keys())
    winners = []
    losers = []

    def run_match(player_names):
        i, j = np.random.choice(range(len(player_names)), size=2, replace=False)
        if compete(player_names[i], player_names[j]):
            return i, j
        return None

    results = Parallel(n_jobs=-1)(
        delayed(run_match)(player_names) for _ in tqdm(range(n_matches))
    )
    for result in results:
        if result is not None:
            winners.append(result[0])
            losers.append(result[1])
    winners = np.array(winners)
    losers = np.array(losers)
    estimates = []
    for _ in tqdm(range(n_bootstrap)):
        matches_bootstrap = np.random.choice(
            range(len(winners)), size=len(winners), replace=True
        )
        winners_bootstrap = winners[matches_bootstrap]
        losers_bootstrap = losers[matches_bootstrap]
        arena = Arena(len(player_names))
        arena.add_data(winners_bootstrap, losers_bootstrap)
        arena.fit()
        scores_estimate = np.round(arena.get_scores(), 2)
        estimates.append(scores_estimate)
    estimates = np.array(estimates)
    means = np.mean(estimates, axis=0)
    stds = np.std(estimates, axis=0)
    print("Scores after training:")
    for name, score, std in sorted(
        zip(player_names, means, stds),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{name}: {score:.2f} ± {std:.2f}")


def test_new_arena():
    # Simple test for the new arena implementation
    winners = np.array([0, 1, 0])
    losers = np.array([1, 2, 2])
    arena = Arena(3)
    arena.add_data(winners, losers)
    arena.fit()

    scores = np.round(arena.get_scores(), 2)
    print("Scores after training:")
    print(scores)

    def w(i, j):
        return arena.predict_win(np.array([i]), np.array([j])).item()

    print("\nWin probabilities:")
    print(f"P(A wins over B): {w(0, 1):.2f}")
    print(f"P(B wins over C): {w(1, 2):.2f}")
    print(f"P(A wins over C): {w(0, 2):.2f}")


if __name__ == "__main__":
    run_ml_arena()
