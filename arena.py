from collections import defaultdict
from typing import Any, Callable

import numpy as np
from beartype import beartype as typed
from openskill import rate, Rating


class Arena:
    @typed
    def __init__(self, compete_fn: Callable[[Any, Any], bool]):
        self.compete_fn = compete_fn
        self.entities = {}  # Maps entity to its Rating

    @typed
    def add_entity(self, entity: Any) -> None:
        if entity not in self.entities:
            self.entities[entity] = Rating()

    @typed
    def run(self, n_matches: int) -> None:
        """Run n matches between randomly selected entities."""
        entities = list(self.entities.keys())
        if len(entities) < 2:
            raise ValueError("Need at least 2 entities to run matches")

        for _ in range(n_matches):
            # Select two different entities randomly
            idx1, idx2 = np.random.choice(len(entities), size=2, replace=False)
            entity1, entity2 = entities[idx1], entities[idx2]

            # Run the match
            entity1_won = self.compete_fn(entity1, entity2)

            # Update ratings
            team1 = [self.entities[entity1]]
            team2 = [self.entities[entity2]]

            if entity1_won:
                (new_team1,), (new_team2,) = rate([team1, team2])
                winner, loser = entity1, entity2
            else:
                (new_team2,), (new_team1,) = rate([team2, team1])
                winner, loser = entity2, entity1

            self.entities[entity1] = new_team1[0]
            self.entities[entity2] = new_team2[0]

    @typed
    def get_scores(self) -> dict[str, float]:
        """Return a dictionary mapping entity names to their scores (mu values)."""
        return {str(entity): rating.mu for entity, rating in self.entities.items()}


def test_arena():
    # Simple compete function based on numeric values
    def compete(x, y):
        return x > y

    arena = Arena(compete)

    # Add some entities
    for i in range(1, 6):
        arena.add_entity(i)

    # Run matches
    arena.run(20)

    # Get scores
    scores = arena.get_scores()
    print("Final scores:")
    for entity, score in sorted(scores.items(), key=lambda x: float(x[0])):
        print(f"Entity {entity}: {score:.2f}")

    # We expect higher entities to have higher scores on average
    # since they win more often


if __name__ == "__main__":
    test_arena()
