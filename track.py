import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype as typed
from jaxtyping import Float, Int
from numpy import ndarray as ND
from scipy.ndimage import gaussian_filter1d


@typed
def parse_timestamp(filename: str) -> datetime:
    match = re.search(r"arena_(\d{8})_(\d{6})\.csv", filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    raise ValueError(f"Invalid filename format: {filename}")


@typed
def load_arena_files() -> tuple[list[datetime], dict[str, dict[str, list[float]]]]:
    """Load all arena CSV files and return timestamps and ratings by method and metric."""
    files = sorted(
        [
            f
            for f in os.listdir("results")
            if f.startswith("arena_") and f.endswith(".csv")
        ]
    )

    timestamps = []
    # Initialize with empty structure: method -> metric -> list of ratings
    ratings_by_method = {}

    for file in files:
        try:
            timestamp = parse_timestamp(file)
            df = pd.read_csv(f"results/{file}", index_col=0)

            timestamps.append(timestamp)

            # Process each method in the file
            for method, row in df.iterrows():
                if method not in ratings_by_method:
                    ratings_by_method[method] = {metric: [] for metric in df.columns}

                # Add ratings for each metric
                for metric in df.columns:
                    ratings_by_method[method][metric].append(float(row[metric]))
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return timestamps, ratings_by_method


@typed
def plot_ratings(
    timestamps: list[datetime], ratings_by_method: dict[str, dict[str, list[float]]]
) -> None:
    """Plot ratings over time for each metric as subplots in a single figure."""
    metrics = list(next(iter(ratings_by_method.values())).keys())

    # Convert timestamps to numerical format for plotting
    x = np.array(
        [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
    )  # minutes

    # Create a figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 12))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for method, metrics_data in ratings_by_method.items():
            if len(metrics_data[metric]) == len(timestamps):
                y_raw = np.array(metrics_data[metric])

                # Apply Gaussian filter to smooth the data
                # Only apply if we have enough data points
                if len(y_raw) > 3:
                    y_smooth = gaussian_filter1d(y_raw, sigma=2.0)
                    ax.plot(x, y_smooth, marker="o", label=method, lw=1, ms=2)

                    # Plot original data as light dots
                    ax.plot(
                        x,
                        y_raw,
                        "o",
                        alpha=0.3,
                        markersize=3,
                        color=ax.lines[-1].get_color(),
                    )
                else:
                    ax.plot(x, y_raw, marker="o", label=method)

        ax.set_title(f"{metric} Ratings")
        ax.set_xlabel("Time (minutes since first run)")
        ax.set_ylabel("Elo Rating")
        ax.grid(True)

    # Add a common legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def main() -> None:
    timestamps, ratings_by_method = load_arena_files()

    if not timestamps:
        print("No arena CSV files found.")
        return

    print(f"Found {len(timestamps)} arena files.")
    plot_ratings(timestamps, ratings_by_method)


if __name__ == "__main__":
    main()
