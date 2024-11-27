import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_single_pareto_front(
    csv_file: str,
    alpha_non_pareto: float = 0.3,
    pareto_alpha: float = 1.0,
    color: str = "blue",
    save_path: str = None
):
    """
    Plots the Pareto front between modularity and functional_enrichment_score
    from a single CSV file, including dashed lines for best individual scores.

    :param csv_file: Path to the CSV file with the data.
                     The CSV must contain columns 'values_0' and 'values_1'.
    :param alpha_non_pareto: Alpha transparency for points not on the Pareto front.
    :param pareto_alpha: Alpha transparency for points on the Pareto front.
    :param color: Color to use for the points.
    :param color: Color for the dashed lines representing best scores.
    :param save_path: If provided, saves the plot to this path.
    """
    # Read the CSV
    data = pd.read_csv(csv_file)

    # Extract the relevant metrics
    values_0 = data['values_0'].values  # Modularity
    values_1 = data['values_1'].values  # Functional Enrichment Score

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
    ax.scatter(values_0[~is_pareto], values_1[~is_pareto], color=color, alpha=alpha_non_pareto, label="No Pareto Front")
    ax.scatter(values_0[is_pareto], values_1[is_pareto], color=color, alpha=pareto_alpha, label="Pareto Front")

    # Add dashed lines
    ax.axhline(y=max_modularity_y, color=color, linestyle="--", label="Coordinates of max points")
    ax.axvline(x=max_modularity_x, color=color, linestyle="--")
    ax.axvline(x=max_enrichment_x, color=color, linestyle="--")
    ax.axhline(y=max_enrichment_y, color=color, linestyle="--")

    # Customize spines (removing top and right)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels and legend
    ax.set_xlabel("Modularity")
    ax.set_ylabel("Functional Enrichment Score")
    ax.set_title("Pareto Front Visualization")
    ax.legend()

    plt.savefig(save_path, format="png", dpi=300)  # Save as PNG with high resolution
    plt.show()

def plot_pareto_from_multiple_csvs(
    csv_files: List[str],
    colors: List[str],
    alpha_non_pareto: float = 0.3,
    pareto_alpha: float = 1.0,
    lines_alpha: float = 0.5,
    save_path: str = None,
):
    """
    Plots the Pareto front between modularity and functional_enrichment_score
    for multiple CSV files, assigning different colors to each dataset.

    :param csv_files: List of paths to the CSV files. Each CSV must have 'values_0' and 'values_1'.
    :param colors: List of colors for points from each CSV file.
    :param alpha_non_pareto: Alpha transparency for points not on the Pareto front.
    :param pareto_alpha: Alpha transparency for points on the Pareto front.
    :param lines_alpha: Alpha transparency for the lines.
    :param save_path: If provided, saves the plot to this path.
    """
    if len(csv_files) != len(colors):
        raise ValueError("The number of CSV files must match the number of colors.")

    _, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)

    for csv_file, color in zip(csv_files, colors):
        # Load data from CSV
        data = pd.read_csv(csv_file)

        # Extract metrics
        values_0 = data['values_0'].values  # Modularity
        values_1 = data['values_1'].values  # Functional Enrichment Score
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
            label=csv_file[22:-4],
        )

        # Add dashed lines
        ax.axhline(y=max_modularity_y, color=color, linestyle="--", alpha=lines_alpha)
        ax.axvline(x=max_modularity_x, color=color, linestyle="--", alpha=lines_alpha)
        ax.axvline(x=max_enrichment_x, color=color, linestyle="--", alpha=lines_alpha)
        ax.axhline(y=max_enrichment_y, color=color, linestyle="--", alpha=lines_alpha)

    # Customize spines (removing top and right)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels and legend
    ax.set_xlabel("Modularity")
    ax.set_ylabel("Functional Enrichment Score")
    ax.set_title("Pareto Front Visualization for Multiple Datasets")
    ax.legend()

    # Save the plot
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)

    plt.show()


# Example usage:
csv_base = "../../results/results_"
csv_file = "../../results/results_multilevel.csv"
#plot_single_pareto_front(csv_file, alpha_non_pareto=0.1, pareto_alpha=1.0, color="purple", save_path="pareto_single.png")

plot_pareto_from_multiple_csvs(
    csv_files=[csv_base+"leiden.csv", csv_base+"multilevel.csv", csv_base+"walktrap.csv"],
    colors=["#1f77b4", "purple", "#2ca02c"],
    alpha_non_pareto=0.2,
    pareto_alpha=1.0,
    lines_alpha=0.5,
    save_path="pareto_comparison.pdf"
)