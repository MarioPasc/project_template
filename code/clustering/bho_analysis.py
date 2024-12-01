#!/usr/bin/env python3

import os
from typing import List

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "300"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

SHOW = False  # For debugging you may set this to true


def plot_single_pareto_front(
    csv_file: str,
    alpha_non_pareto: float = 0.3,
    pareto_alpha: float = 1.0,
    lines_alpha: float = 0.5,
    color: str = "blue",
    save_path: str = None,
    title: str = "",
):
    """
    Plots the Pareto front between modularity and functional_enrichment_score
    from a single CSV file, including dashed lines for best individual scores.

    :param csv_file: Path to the CSV file with the data.
                     The CSV must contain columns 'values_0' and 'values_1'.
    :param alpha_non_pareto: Alpha transparency for points not on the Pareto front.
    :param pareto_alpha: Alpha transparency for points on the Pareto front.
    :param lines_alpha: Alpha transparency for the lines.
    :param color: Color to use for the points.
    :param color: Color for the dashed lines representing best scores.
    :param save_path: If provided, saves the plot to this path.
    :param title: Title for the plot.
    """
    # Read the CSV
    data = pd.read_csv(csv_file)

    # Extract the relevant metrics
    values_0 = data["values_0"].values  # Modularity
    values_1 = data["values_1"].values  # Functional Enrichment Score

    # Combine the values into pairs
    points = np.column_stack((values_0, values_1))

    # Determine Pareto front
    is_pareto = np.zeros(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        dominated = np.any(
            np.logical_and(points[:, 0] >= point[0], points[:, 1] >= point[1])
            & np.logical_not(np.all(points[:, 0:2] == point[0:2], axis=1))
        )
        if not dominated:
            is_pareto[i] = True

    # Find the index of the highest metrics value
    max_modularity_index = np.argmax(values_0)
    max_enrichment_index = np.argmax(values_1)

    # Find the best individual values for vertical/horizontal lines
    max_modularity_x = values_0[max_modularity_index]
    max_modularity_y = values_1[max_modularity_index]

    max_enrichment_x = values_0[max_enrichment_index]
    max_enrichment_y = values_1[max_enrichment_index]

    # Plot
    _, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.scatter(
        values_0[~is_pareto],
        values_1[~is_pareto],
        color=color,
        alpha=alpha_non_pareto,
        label="No Pareto Front",
    )
    ax.scatter(
        values_0[is_pareto],
        values_1[is_pareto],
        color=color,
        alpha=pareto_alpha,
        label="Pareto Front",
    )

    # Add dashed lines
    ax.axhline(
        y=max_modularity_y,
        color=color,
        linestyle="--",
        label="Coordinates of max points",
        alpha=lines_alpha,
    )
    ax.axvline(x=max_modularity_x, color=color, linestyle="--", alpha=lines_alpha)
    ax.axvline(x=max_enrichment_x, color=color, linestyle="--", alpha=lines_alpha)
    ax.axhline(y=max_enrichment_y, color=color, linestyle="--", alpha=lines_alpha)

    # Customize spines (removing top and right)
    ax.spines[["right", "top"]].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Labels and legend
    ax.set_xlabel("Modularity")
    ax.set_ylabel("Functional Enrichment Score")
    ax.set_title(title)
    ax.legend()

    plt.savefig(save_path, format="pdf", dpi=300)  # Save as PNG with high resolution
    if SHOW:
        plt.show()


def plot_pareto_from_multiple_csvs(
    csv_files: List[str],
    legend_names: List[str],
    colors: List[str],
    alpha_non_pareto: float = 0.3,
    pareto_alpha: float = 1.0,
    lines_alpha: float = 0.5,
    save_path: str = None,
    title: str = "",
):
    """
    Plots the Pareto front between modularity and functional_enrichment_score
    for multiple CSV files, assigning different colors to each dataset.

    :param csv_files: List of paths to the CSV files. Each CSV must have 'values_0' and 'values_1'.
    :param legend_names: List of names for the legend corresponding to each CSV file.
    :param colors: List of colors for points from each CSV file.
    :param alpha_non_pareto: Alpha transparency for points not on the Pareto front.
    :param pareto_alpha: Alpha transparency for points on the Pareto front.
    :param lines_alpha: Alpha transparency for the lines.
    :param save_path: If provided, saves the plot to this path.
    :param title: Title for the plot.
    """
    if len(csv_files) != len(colors) or len(csv_files) != len(legend_names):
        raise ValueError(
            "The number of CSV files, colors, and legend names must match."
        )

    _, ax = plt.subplots(figsize=(10, 6))

    for idx, (csv_file, color, legend_name) in enumerate(
        zip(csv_files, colors, legend_names)
    ):
        # Load data from CSV
        data = pd.read_csv(csv_file)

        # Extract metrics
        values_0 = data["values_0"].values  # Modularity
        values_1 = data["values_1"].values  # Functional Enrichment Score
        points = np.column_stack((values_0, values_1))

        # Determine Pareto front
        is_pareto = np.zeros(points.shape[0], dtype=bool)
        for i, point in enumerate(points):
            dominated = np.any(
                np.logical_and(points[:, 0] >= point[0], points[:, 1] >= point[1])
                & np.logical_not(np.all(points[:, 0:2] == point[0:2], axis=1))
            )
            if not dominated:
                is_pareto[i] = True

        # Sort Pareto points for the connecting line
        pareto_points = points[is_pareto]
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

        # Find the best individual values for vertical/horizontal lines
        max_modularity_x = pareto_points[-1, 0]
        max_modularity_y = pareto_points[-1, 1]

        max_enrichment_x = pareto_points[0, 0]
        max_enrichment_y = pareto_points[0, 1]

        # Plot the points
        ax.scatter(
            values_0[~is_pareto],
            values_1[~is_pareto],
            color=color,
            alpha=alpha_non_pareto,
        )
        ax.scatter(
            values_0[is_pareto],
            values_1[is_pareto],
            color=color,
            alpha=pareto_alpha,
            label=legend_name,
        )

        # Plot the Pareto line
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            color=color,
            alpha=pareto_alpha,
            linestyle="-",
            linewidth=1.5,
        )
        ax.set_ylim([0.05, 0.25])
        ax.set_xlim([0.2, 0.3])
        # Calculate normalized positions for lines
        x_norm_modularity = (max_modularity_x - ax.get_xlim()[0]) / (
            ax.get_xlim()[1] - ax.get_xlim()[0]
        )
        y_norm_modularity = (max_modularity_y - ax.get_ylim()[0]) / (
            ax.get_ylim()[1] - ax.get_ylim()[0]
        )

        x_norm_enrichment = (max_enrichment_x - ax.get_xlim()[0]) / (
            ax.get_xlim()[1] - ax.get_xlim()[0]
        )
        y_norm_enrichment = (max_enrichment_y - ax.get_ylim()[0]) / (
            ax.get_ylim()[1] - ax.get_ylim()[0]
        )

        # Add dashed lines stopping exactly at the points
        ax.axhline(
            y=max_modularity_y,
            xmin=0,
            xmax=x_norm_modularity,
            color=color,
            linestyle="--",
            alpha=lines_alpha,
        )
        ax.axvline(
            x=max_modularity_x,
            ymin=0,
            ymax=y_norm_modularity,
            color=color,
            linestyle="--",
            alpha=lines_alpha,
        )
        ax.axvline(
            x=max_enrichment_x,
            ymin=0,
            ymax=y_norm_enrichment,
            color=color,
            linestyle="--",
            alpha=lines_alpha,
        )
        ax.axhline(
            y=max_enrichment_y,
            xmin=0,
            xmax=x_norm_enrichment,
            color=color,
            linestyle="--",
            alpha=lines_alpha,
        )

    # Customize spines (removing top and right)
    ax.spines[["right", "top"]].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Labels and legend
    ax.set_xlabel("Modularidad (Q)")
    ax.set_ylabel("Coeficiente de Significancia Biológica (QBS)")
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=len(csv_files))

    # Save the plot
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    if SHOW:
        plt.show()


def plot_hyperparameter_vs_metric_fixed_hyperparam_subplots(
    csv_files: List[str],
    legend_names: List[str],
    metric_column: dict,
    colors: List[str] = ["blue", "orange", "green"],
    max_line_colors: List[str] = ["black", "black"],
    alpha: float = 0.7,
    lines_alpha: float = 0.5,
    save_path: str = None,
    label_x: str = "",
    title: str = "",
):
    """
    Plots the chosen metrics against the hyperparameter (assumed to be the third column)
    from multiple CSV files. Creates subplots for each metric.

    :param csv_files: List of CSV file paths.
    :param legend_names: List of names for the legend corresponding to each CSV file.
    :param metric_column: Dictionary with keys as column names in the CSV (metrics) and values as display names of the metrics.
    :param colors: List of colors to use for each CSV file.
    :param max_line_colors: List of colors to use for the maximum metric lines.
    :param alpha: Alpha transparency for the scatter points.
    :param lines_alpha: Alpha transparency for the dashed lines indicating max metric values.
    :param save_path: Path to save the plot. If None, the plot is just displayed.
    :param label_x: Label for the X-axis (e.g., "Hyperparameter Value").
    :param title: Title for the entire plot.
    """
    num_metrics = len(metric_column)
    if num_metrics > 2:
        raise ValueError("Only up to 2 metrics are supported for subplots.")

    fig, axes = plt.subplots(1, num_metrics, figsize=(12, 5), sharex=True)
    if num_metrics == 1:
        axes = [axes]  # Make axes iterable even for a single subplot

    max_metric_lines = []  # To store legend entries for max lines

    for metric_idx, (metric_key, metric_name) in enumerate(metric_column.items()):
        ax = axes[metric_idx]
        max_value_overall = -np.inf
        max_model_name = ""
        max_metric_value = -np.inf

        for idx, csv_file in enumerate(csv_files):
            # Load the CSV
            data = pd.read_csv(csv_file)

            # Automatically detect the hyperparameter column (third column)
            hyperparameter_column = data.columns[3]
            x_values = data[hyperparameter_column]
            y_values = data[metric_key]

            # Scatter plot for the current dataset
            color = colors[idx % len(colors)]
            ax.scatter(
                x_values, y_values, label=legend_names[idx], color=color, alpha=alpha
            )

            # Check if this model has the max value for the metric
            current_max = y_values.max()
            if current_max > max_metric_value:
                max_metric_value = current_max
                max_model_name = legend_names[idx]
                max_value_overall = current_max

        # Add a dashed line for the maximum value
        max_line_color = max_line_colors[metric_idx % len(max_line_colors)]
        ax.axhline(
            y=max_value_overall, color=max_line_color, linestyle="--", alpha=lines_alpha
        )
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlim([0.0, 2.1])
        ax.set_ylim([-0.01, 0.35])
        # Add to legend entries for max lines
        max_metric_lines.append(
            f"Max {metric_name} ({max_model_name}: {max_value_overall:.2f})"
        )

        # Axis labels
        ax.set_xlabel(label_x)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}")

        # Customize spines
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(False)

    import matplotlib.lines as mlines

    # Create custom legend handles for max metric lines
    max_metric_handles = [
        mlines.Line2D(
            [],
            [],
            color=max_line_colors[i % len(max_line_colors)],
            linestyle="--",
            label=label,
        )
        for i, label in enumerate(max_metric_lines)
    ]

    # Get scatter plot handles and labels
    scatter_handles, scatter_labels = axes[0].get_legend_handles_labels()

    # Combine scatter handles and max metric handles
    all_handles = scatter_handles + max_metric_handles
    all_labels = scatter_labels + max_metric_lines

    # Add the legend to the figure
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(csv_files),
        fontsize=9,
    )

    # Overall title
    fig.suptitle(title)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    if SHOW:
        plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Visual analysis of optimization results")
    
    parser.add_argument(
        "results_folder",
        type=str,
        help="Path to the file containing the results of the optimization (e.g. ./results/)",
    )
    
    args = parser.parse_args()

    folder_path = args.results_folder

    plots_folder: os.PathLike = os.path.join(folder_path, "plots/optimization")
    os.makedirs(plots_folder, exist_ok=True)

    # By this time we can find the results_*.csv files in the results/ folder. 
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    # List all files in the folder and filter for 'results_*.csv'
    # These files are the results of the optimization process.
    csv_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.startswith("results_") and file.endswith(".csv")
    ]

    colors = ["#0C5DA5", "#00B945"]

    plot_pareto_from_multiple_csvs(
        csv_files=csv_files,
        legend_names=["Leiden", "Multilevel"],
        colors=colors,
        alpha_non_pareto=0.2,
        pareto_alpha=1.0,
        lines_alpha=0.5,
        save_path=os.path.join(plots_folder, "pareto_comparison.pdf"),
        title="",
    )

    plot_hyperparameter_vs_metric_fixed_hyperparam_subplots(
        csv_files=csv_files,
        legend_names=["Leiden", "Multilevel"],
        metric_column={
            "values_0": "Modularidad (Q)",
            "values_1": "Coeficiente de Significancia Biológica (QBS)",
        },  # Eje Y
        colors=colors,
        alpha=0.8,
        lines_alpha=0.5,
        save_path=os.path.join(plots_folder, "hyperparameter_vs_metric.pdf"),
        label_x=r"Resolución ($\gamma$)",
        title="",
    )

if __name__ == "__main__":
    main()
