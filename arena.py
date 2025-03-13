from collections import defaultdict
from typing import Any, Callable

import numpy as np
from beartype import beartype as typed
from jaxtyping import Float
from openskill.models import BradleyTerryFull
from sklearn.linear_model import ARDRegression, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm


class Arena:
    @typed
    def __init__(self, compete_fn: Callable[[Any, Any], bool]):
        self.compete_fn = compete_fn
        self.entities = {}  # Maps entity to its Rating
        self.model = BradleyTerryFull()

    @typed
    def add_entity(self, entity: str) -> None:
        if entity not in self.entities:
            self.entities[entity] = self.model.rating(name=entity)

    @typed
    def run(self, n_matches: int) -> None:
        """Run n matches between randomly selected entities."""
        entities = list(self.entities.keys())
        if len(entities) < 2:
            raise ValueError("Need at least 2 entities to run matches")
        for _ in tqdm(range(n_matches)):
            idx1, idx2 = np.random.choice(len(entities), size=2, replace=False)
            entity1, entity2 = entities[idx1], entities[idx2]

            entity1_won = self.compete_fn(entity1, entity2)

            team1 = [self.entities[entity1]]
            team2 = [self.entities[entity2]]
            if entity1_won:
                (new_team1,), (new_team2,) = self.model.rate([team1, team2])
                winner, loser = entity1, entity2
            else:
                (new_team2,), (new_team1,) = self.model.rate([team2, team1])
                winner, loser = entity2, entity1

            self.entities[entity1] = new_team1
            self.entities[entity2] = new_team2

    @typed
    def get_scores(self) -> dict[str, float]:
        """Return a dictionary mapping entity names to their scores (mu values)."""
        return {str(entity): rating.mu for entity, rating in self.entities.items()}

    @typed
    def predict_win(self, x: str, y: str) -> float:
        """Predict the probability of x winning over y."""
        x_rating = self.entities[x]
        y_rating = self.entities[y]
        return self.model.predict_win([[x_rating], [y_rating]])[0]


@typed
def generate_dataset(
    outliers: bool = False,
    bad_features: bool = False,
    n_samples: int = 1000,
    n_features: int = 20,
) -> tuple[
    Float[np.ndarray, "n_samples n_features"],
    Float[np.ndarray, "n_samples"],
    Float[np.ndarray, "n_test n_features"],
    Float[np.ndarray, "n_test"],
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


@typed
def create_player_list() -> dict[str, Callable]:
    """Create a list of ML models to compare."""
    players = {}

    # Ridge with adaptive alpha
    def ridge_adaptive(X_train, y_train, X_test):
        alpha = 1.0 / (2 * len(X_train))
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    players["Ridge(adaptive)"] = ridge_adaptive

    # Ridge with fixed alpha
    def ridge_fixed(X_train, y_train, X_test):
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    players["Ridge(1.0)"] = ridge_fixed

    # ARDRegression
    def ard(X_train, y_train, X_test):
        model = ARDRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    players["ARDRegression"] = ard

    # LinearRegression
    def linear(X_train, y_train, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    players["LinearRegression"] = linear

    return players


def test_ml_arena():
    players = create_player_list()

    # Create a competition function
    def compete(player1, player2):
        # Generate a dataset
        # outliers = np.random.random() > 0.5
        # bad_features = np.random.random() > 0.5
        X_train, y_train, X_test, y_test = generate_dataset(
            outliers=False, bad_features=True
        )

        # Get predictions
        pred1 = players[player1](X_train, y_train, X_test)
        pred2 = players[player2](X_train, y_train, X_test)

        # Calculate MSE
        mse1 = mean_squared_error(y_test, pred1)
        mse2 = mean_squared_error(y_test, pred2)

        # Lower MSE is better
        return mse1 < mse2

    # Create arena
    arena = Arena(compete)

    # Add players
    for player_name in players:
        arena.add_entity(player_name)

    # Run matches
    arena.run(5000)

    # Print results
    print("\nFinal scores:")
    scores = arena.get_scores()
    for player, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{player}: {score:.2f}")

    # Print win probabilities
    print("\nWin probabilities:")
    player_names = list(players.keys())
    for i in range(len(player_names)):
        for j in range(i + 1, len(player_names)):
            p1, p2 = player_names[i], player_names[j]
            prob = arena.predict_win(p1, p2)
            print(f"P({p1} wins over {p2}): {prob:.2f}")


if __name__ == "__main__":
    test_ml_arena()
