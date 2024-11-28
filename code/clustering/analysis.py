import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os
import scienceplots

plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['font.size'] = 10
plt.rcParams.update({'figure.dpi': '300'})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

SHOW = False # For debugging you may set this to true

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
    ax.axhline(y=max_modularity_y, color=color, linestyle="--", label="Coordinates of max points", alpha=lines_alpha)
    ax.axvline(x=max_modularity_x, color=color, linestyle="--", alpha=lines_alpha)
    ax.axvline(x=max_enrichment_x, color=color, linestyle="--", alpha=lines_alpha)
    ax.axhline(y=max_enrichment_y, color=color, linestyle="--", alpha=lines_alpha)

    # Customize spines (removing top and right)
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Labels and legend
    ax.set_xlabel("Modularity")
    ax.set_ylabel("Functional Enrichment Score")
    ax.set_title(title)
    ax.legend()

    plt.savefig(save_path, format="pdf", dpi=300)  # Save as PNG with high resolution
    if SHOW: plt.show()

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
        raise ValueError("The number of CSV files, colors, and legend names must match.")

    _, ax = plt.subplots(figsize=(10, 6))

    for idx, (csv_file, color, legend_name) in enumerate(zip(csv_files, colors, legend_names)):
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
        ax.set_ylim([0.0, 0.2])
        ax.set_xlim([0.0, 0.2])
        # Calculate normalized positions for lines
        x_norm_modularity = (max_modularity_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_norm_modularity = (max_modularity_y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

        x_norm_enrichment = (max_enrichment_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_norm_enrichment = (max_enrichment_y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

        # Add dashed lines stopping exactly at the points
        ax.axhline(y=max_modularity_y, xmin=0, xmax=x_norm_modularity, color=color, linestyle="--", alpha=lines_alpha)
        ax.axvline(x=max_modularity_x, ymin=0, ymax=y_norm_modularity, color=color, linestyle="--", alpha=lines_alpha)
        ax.axvline(x=max_enrichment_x, ymin=0, ymax=y_norm_enrichment, color=color, linestyle="--", alpha=lines_alpha)
        ax.axhline(y=max_enrichment_y, xmin=0, xmax=x_norm_enrichment, color=color, linestyle="--", alpha=lines_alpha)


    # Customize spines (removing top and right)
    ax.spines[['right', 'top']].set_visible(False)
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

    if SHOW: plt.show()

def plot_hyperparameter_vs_metric_fixed_hyperparam(
    csv_files: List[str],
    legend_names: List[str],
    metric_column: str,
    colors: List[str] = ["blue", "orange", "green"],
    alpha: float = 0.7,
    lines_alpha: float = 0.5,
    save_path: str = None,
    label_x: str = "",
    label_y: str = "",
    title: str = "",
):
    """
    Plots the chosen metric (values_0 or values_1) against the hyperparameter 
    (assumed to be the third column) from multiple CSV files.

    :param csv_files: List of CSV file paths.
    :param legend_names: List of names for the legend corresponding to each CSV file.
    :param metric_column: Column name of the metric to be plotted on the Y-axis (e.g., 'values_0' or 'values_1').
    :param colors: List of colors to use for each CSV file.
    :param alpha: Alpha transparency for the scatter points.
    :param lines_alpha: Alpha transparency for the dashed lines indicating max metric values.
    :param save_path: Path to save the plot. If None, the plot is just displayed.
    :param label_x: Label for the X-axis (e.g., "Hyperparameter Value").
    :param label_y: Label for the Y-axis (e.g., "Metric Value").
    :param title: Title for the plot.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    
    for idx, csv_file in enumerate(csv_files):
        # Load the CSV
        data = pd.read_csv(csv_file)
        
        # Automatically detect the hyperparameter column (third column)
        hyperparameter_column = data.columns[2]
        x_values = data[hyperparameter_column]
        y_values = data[metric_column]
        
        # Scatter plot for the current dataset
        color = colors[idx % len(colors)]  # Cycle through colors if not enough provided
        max_modularity = max(y_values)
        ax.axhline(y=max_modularity, color=color, linestyle="--", alpha=lines_alpha)
        ax.scatter(x_values, y_values, label=legend_names[idx], color=color, alpha=alpha)
    
    # Add labels and legend
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    # Customize spines (removing top and right)
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)
    if SHOW: plt.show()




if __name__ == "__main__":
    # Example usage:
    csv_base = "../../results/results_"
    csv_file = "../../results/results_multilevel.csv"
    csv_files = [csv_base+"leiden.csv", csv_base+"multilevel.csv"]
    colors = ['#0C5DA5', '#00B945']
    results_path = "../../results"

    plot_single_pareto_front(
        csv_file,
        alpha_non_pareto=0.1,
        pareto_alpha=1.0,
        lines_alpha=0.5,
        color="purple",
        save_path=os.path.join(results_path, "pareto_single.pdf"),
        title="Pareto Front")


    plot_pareto_from_multiple_csvs(
        csv_files=csv_files,
        legend_names=["Leiden", "Multilevel"],
        colors=colors,
        alpha_non_pareto=0.2,
        pareto_alpha=1.0,
        lines_alpha=0.5,
        save_path=os.path.join(results_path, "pareto_comparison.pdf"),
        title=""
    )


    plot_hyperparameter_vs_metric_fixed_hyperparam(
        csv_files=csv_files,
        legend_names=["Leiden", "Multilevel", "Walktrap"],
        metric_column="values_0",  # Eje Y
        colors=colors,
        alpha=0.8,
        lines_alpha=0.5,
        save_path=os.path.join(results_path, "hyperparameter_vs_metric.pdf"),
        label_x="Hyperparameter Value",
        label_y="Modularity",
        title="Modularity values vs Model's hyperparameter values",
    )
