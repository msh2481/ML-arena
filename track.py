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


@typed
def parse_timestamp(filename: str) -> datetime:
    match = re.search(r"arena_(\d{8})_(\d{6})\.csv", filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    raise ValueError(f"Invalid filename format: {filename}")


@typed
def parse_metrics_filename(filename: str) -> tuple[datetime, str]:
    """Parse metrics filename to extract timestamp and dataset name."""
    match = re.search(r"metrics_(\d{8})_(\d{6})_(.+)\.csv", filename)
    if match:
        date_str, time_str, dataset = match.groups()
        timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return timestamp, dataset
    raise ValueError(f"Invalid metrics filename format: {filename}")


@typed
def load_metrics_files() -> (
    tuple[list[datetime], dict[str, dict[str, dict[str, list[float]]]]]
):
    """Load all metrics CSV files and return timestamps, metrics by dataset, method, and metric."""
    files = sorted(
        [
            f
            for f in os.listdir("results")
            if f.startswith("metrics_") and f.endswith(".csv")
        ]
    )

    timestamps = []
    # Initialize with empty structure: dataset -> method -> metric -> list of values
    metrics_by_dataset = {}

    for file in files:
        try:
            timestamp, dataset = parse_metrics_filename(file)
            df = pd.read_csv(f"results/{file}", index_col=0)

            timestamps.append(timestamp)

            # Initialize dataset if not seen before
            if dataset not in metrics_by_dataset:
                metrics_by_dataset[dataset] = {}

            # Process each method in the file
            for method, row in df.iterrows():
                if method not in metrics_by_dataset[dataset]:
                    metrics_by_dataset[dataset][method] = {
                        metric: [] for metric in df.columns
                    }

                # Add metrics for each metric type
                for metric in df.columns:
                    metrics_by_dataset[dataset][method][metric].append(
                        float(row[metric])
                    )
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return timestamps, metrics_by_dataset


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
def compute_running_stats(
    data: Float[ND, "n"], window_size: int = 5
) -> tuple[Float[ND, "n"], Float[ND, "n"]]:
    """Compute running mean and standard deviation with specified window size."""
    n = len(data)
    running_mean = np.zeros_like(data)
    running_std = np.zeros_like(data)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = data[start : i + 1]
        running_mean[i] = np.mean(window)
        running_std[i] = (
            np.std(window, ddof=1) / np.sqrt(len(window)) if len(window) > 1 else 0
        )

    return running_mean, running_std


@typed
def plot_metrics(
    timestamps: list[datetime],
    metrics_by_dataset: dict[str, dict[str, dict[str, list[float]]]],
) -> None:
    """Plot metrics over time for each metric and dataset as subplots in separate figures."""
    # Get all unique metrics across all datasets
    all_metrics = set()
    for dataset_data in metrics_by_dataset.values():
        for method_data in dataset_data.values():
            all_metrics.update(method_data.keys())

    all_metrics = sorted(all_metrics)

    # For each metric, create a figure with subplots for each dataset
    for metric in all_metrics:
        # Count datasets that have this metric
        datasets_with_metric = [
            dataset
            for dataset, dataset_data in metrics_by_dataset.items()
            if any(metric in method_data for method_data in dataset_data.values())
        ]

        if not datasets_with_metric:
            continue

        n_datasets = len(datasets_with_metric)
        n_cols = (n_datasets + 2) // 3
        fig, axes = plt.subplots(3, n_cols, figsize=(30, 20))
        if n_datasets == 1:
            axes = [axes]  # Make it iterable for the loop

        for i, dataset in enumerate(datasets_with_metric):
            # Get the appropriate subplot for this dataset
            if n_datasets == 1:
                ax = axes
            else:
                # For multiple datasets, we have a 3-row grid
                ax_row = i // n_cols
                ax_col = i % n_cols
                ax = axes[ax_row, ax_col] if n_datasets > 3 else axes[i]
            dataset_data = metrics_by_dataset[dataset]

            x = np.array(timestamps[: len(next(iter(dataset_data.values()))[metric])])

            for method in dataset_data:
                method_data = dataset_data[method]
                y_raw = np.array(method_data[metric])

                if len(y_raw) != len(x):
                    # Skip if data lengths don't match
                    continue

                mean_metric = float(np.mean(y_raw))
                std_metric = float(np.std(y_raw, ddof=1) / np.sqrt(len(y_raw)))
                label = f"{method} ({mean_metric:.4f}±{std_metric:.4f})"

                # Only apply running stats if we have enough data points
                if len(y_raw) > 3:
                    # Compute running mean and standard deviation
                    y_mean, y_std = compute_running_stats(y_raw, window_size=5)

                    # Plot the mean line
                    (line,) = ax.plot(x, y_mean, label=label, lw=2)
                    color = line.get_color()

                    # Plot the standard deviation band
                    ax.fill_between(
                        x, y_mean - y_std, y_mean + y_std, alpha=0.02, color=color
                    )

                    # Plot original data as light dots
                    ax.plot(
                        x,
                        y_raw,
                        "o",
                        alpha=0.3,
                        markersize=3,
                        color=color,
                    )
                else:
                    ax.plot(x, y_raw, marker="o", label=label)

            ax.set_title(f"{dataset.replace('_', ' ').title()}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{metric} Value")
            ax.grid(True)
            ax.legend(loc="best")

        plt.suptitle(f"{metric} Metrics Across Datasets", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"metrics_{metric}.png")
        plt.close()


@typed
def plot_ratings(
    timestamps: list[datetime], ratings_by_method: dict[str, dict[str, list[float]]]
) -> None:
    """Plot ratings over time for each metric as subplots in a single figure."""
    metrics = list(next(iter(ratings_by_method.values())).keys())
    x = np.array(timestamps)
    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 12))
    methods_with_mse_ratings = []
    for method, metrics_data in ratings_by_method.items():
        y_raw = metrics_data["MSE"]
        mean_rating = int(np.mean(y_raw))
        methods_with_mse_ratings.append((mean_rating, method))
    methods_with_mse_ratings.sort(reverse=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for _, method in methods_with_mse_ratings:
            metrics_data = ratings_by_method[method]
            if len(metrics_data[metric]) == len(timestamps):
                y_raw = np.array(metrics_data[metric])
                mean_rating = int(np.mean(y_raw))
                std_rating = int(np.std(y_raw, ddof=1) / np.sqrt(len(y_raw)))
                label = f"{method} ({mean_rating}±{std_rating})"

                # Only apply if we have enough data points
                if len(y_raw) > 3:
                    # Compute running mean and standard deviation
                    y_mean, y_std = compute_running_stats(y_raw, window_size=5)

                    # Plot the mean line
                    (line,) = ax.plot(x, y_mean, label=label, lw=2)
                    color = line.get_color()

                    # Plot the standard deviation band
                    ax.fill_between(
                        x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color
                    )

                    # Plot original data as light dots
                    ax.plot(
                        x,
                        y_raw,
                        "o",
                        alpha=0.3,
                        markersize=3,
                        color=color,
                    )
                else:
                    ax.plot(x, y_raw, marker="o", label=label)

        ax.set_title(f"{metric} Ratings")
        ax.set_xlabel("Time")
        ax.set_ylabel("Elo Rating")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0)
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def main() -> None:
    arena_timestamps, ratings_by_method = load_arena_files()
    if arena_timestamps:
        print(f"Found {len(arena_timestamps)} arena files.")
        plot_ratings(arena_timestamps, ratings_by_method)
    else:
        print("No arena CSV files found.")

    metrics_timestamps, metrics_by_dataset = load_metrics_files()
    if metrics_timestamps:
        print(f"Found {len(metrics_timestamps)} metrics files.")
        plot_metrics(metrics_timestamps, metrics_by_dataset)
    else:
        print("No metrics CSV files found.")


if __name__ == "__main__":
    main()
