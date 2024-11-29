#!/usr/bin/env python3

# The main goal of this analysis is to execute the algorithms with the best hyperparameters that we got from the Bayesian Optimization
# To archieve this goal, we must:
#   1. Read the csv files generated in results/ and get the two most extreme pareto configurations.
#   2. Execute our baseline (Fast Greedy) and also plot its results.
#   3. Use the network/network.visualize_clusters() function to visualize the clustering results. 
#   4. Create a 5-subplot visualization of each algorithm result.
#   5. (Maybe) Encapsulate the functionality of selecting an specific configuration to plot.

import os
from typing import Union, Dict, Tuple, List, Any

import pandas as pd
import numpy as np
from igraph import Graph
import shutil
import argparse
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
import scienceplots

# Style
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "300"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

from algorithms import Algorithms
from network import network
from utils import misc
from metrics import Metrics

def get_extreme_configurations(csv_path: Union[str, os.PathLike]) -> Dict[Tuple[float, float], Tuple[float, str]]:
    """
    Extract extreme configurations from the Pareto front, maximizing values_0 and values_1,
    and dynamically identify the relevant params_* column.

    Args:
        csv_path (Union[str, os.PathLike]): Path to the CSV file containing the dataset.

    Returns:
        Dict[Tuple[float, float], Tuple[float, str]]: A dictionary where keys are tuples of extreme (values_0, values_1)
                                                      and values are tuples of (hyperparameter value, column name).
    """
    # Load the dataset
    data: pd.DataFrame = pd.read_csv(csv_path)

    # Extract the relevant metrics
    values_0 = data["values_0"].values
    values_1 = data["values_1"].values

    # Dynamically find the params_* column
    params_columns = [col for col in data.columns if col.startswith("params_")]
    if len(params_columns) != 1:
        raise ValueError("Expected exactly one 'params_*' column in the dataset, found: {}".format(params_columns))
    params_column = params_columns[0]

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

    # Filter the Pareto-efficient points
    pareto_points = data[is_pareto]

    # Find the index of the highest metric values
    max_modularity_index = pareto_points["values_0"].idxmax()
    max_enrichment_index = pareto_points["values_1"].idxmax()

    # Get the extreme configurations and their params values
    extremes: Dict[Tuple[float, float], Tuple[float, str]] = {
        (data.loc[max_modularity_index, "values_0"], data.loc[max_modularity_index, "values_1"]): (
            data.loc[max_modularity_index, params_column],
            params_column,
        ),
        (data.loc[max_enrichment_index, "values_0"], data.loc[max_enrichment_index, "values_1"]): (
            data.loc[max_enrichment_index, params_column],
            params_column,
        ),
    }

    return extremes

def plot_saved_clustering_results(saved_clustering_paths: Dict[str, Dict[float, Tuple[str, str]]]) -> None:
    """
    Display saved clustering plots for a single algorithm with columns as parameter values.

    Args:
        saved_clustering_paths (Dict[str, Dict[float, Tuple[str, str]]]): 
            Dictionary of clustering image paths and parameter names for one algorithm.
            The inner value is a tuple of (image_path, parameter_name).
    """
    # There should only be one algorithm in this dictionary
    algorithm_name = next(iter(saved_clustering_paths.keys()))
    param_data = saved_clustering_paths[algorithm_name]

    num_params = len(param_data)

    # Create a figure with subplots (1 row for the single algorithm)
    fig, axs = plt.subplots(
        1, num_params,
        figsize=(5 * num_params, 5),
        constrained_layout=True
    )

    # Ensure axs is always iterable
    if num_params == 1:
        axs = [axs]

    # Define parameter name mappings
    param_name_map = {
        "params_resolution": r"Resolución $(\gamma)$",
        "params_steps": "Pasos (Steps)",
    }

    # Define algorithm name mappings
    algorithm_name_map = {
        "multilevel": "Louvain",
        "leiden": "Leiden",
        "walktrap": "Walktrap",
        "fastgreedy": "Fast Greedy"
    }

    # Get display name for the algorithm
    algorithm_display_name = algorithm_name_map.get(algorithm_name, algorithm_name)

    # Loop through parameter values
    for col_idx, (param_value, (img_path, param_name)) in enumerate(param_data.items()):
        # Read and display the saved image
        img = imread(img_path)
        axs[col_idx].imshow(img)
        
        # Get the parameter display name
        param_display_name = param_name_map.get(param_name, param_name)
        
        # Set title with algorithm name, parameter value, and parameter display name
        axs[col_idx].set_title(
            f"{algorithm_display_name} ({param_display_name}: {param_value:.2f})"
        )
        axs[col_idx].axis("off")  # Turn off axes for cleaner display

    # Save the plot for this algorithm
    output_file = f"results/{algorithm_display_name.lower()}_clustering.pdf"
    plt.savefig(output_file)
    plt.close(fig)  # Close the figure to free memory


def main() -> None:
    
    # Setup logger
    os.makedirs("./logs", exist_ok=True)
    logger = misc.setup_logger(
        name="Bayesian_Hyperparameter_Optimization_Clustering_Networks",
        log_file="logs/bho_optimization.log",
    )
    
    parser = argparse.ArgumentParser(description="Visual analysis of clustering results")
    
    parser.add_argument(
        "network",
        type=str,
        help="Path to the file containing the network (e.g., data/genes.tsv)",
    )
    
    parser.add_argument(
        "results_folder",
        type=str,
        help="Path to the file containing the results of the optimization (e.g. ./results/)",
    )
    
    args = parser.parse_args()
    
    folder_path: str = args.results_folder
    
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
    
    # Now, we can extract the most extreme configurations of the clustering algorithm
    configurations: Dict[str, Dict[Tuple[float, float], Tuple[float, str]]] = {}
    for csv_file in csv_files:
        algorithm_name: str = os.path.basename(csv_file).replace("results_", "").replace(".csv", "")
        configurations[algorithm_name] = get_extreme_configurations(csv_path=csv_file)

    defined_algorithms: List[str] = ["walktrap", "leiden", "multilevel"]

    # Check if all keys in configurations are in defined_algorithms
    if not all(algorithm in defined_algorithms for algorithm in configurations.keys()):
        raise ValueError(
            f"You have included the results of an algorithm that is not provided. "
            f"Please, stick to the pipeline. Found: {list(configurations.keys())}"
        )

    graph: Graph = misc.network_to_igraph_format(network_csv = args.network)

    network_worker: network.Network = network.Network(graph=graph)

    # Folder to save clustering plots
    output_folder: str = "clustering_plots"
    output_dir: os.PathLike = os.path.join(args.results_folder, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store saved image paths for each clustering result
    saved_clustering_paths: Dict[str, Dict[float, str]] = {}

    # Loop through each algorithm and its extreme configurations
    for algorithm_name, extreme_configurations in configurations.items():
        if algorithm_name not in saved_clustering_paths:
            saved_clustering_paths[algorithm_name] = {}
        
        for (values_0, values_1), (param_value, param_name) in extreme_configurations.items():
            # Run the appropriate clustering algorithm
            if algorithm_name == "leiden":
                results: List[List[int]] = Algorithms.leiden_clustering(
                    graph=graph, resolution=param_value, logger=logger
                )
            elif algorithm_name == "multilevel":
                results: List[List[int]] = Algorithms.multilevel_clustering(
                    graph=graph, resolution=param_value, logger=logger
                )
            elif algorithm_name == "walktrap":
                results: List[List[int]] = Algorithms.walktrap_clustering(
                    graph=graph, steps=int(param_value), logger=logger
                )

            # Define the output file path
            output_path: os.PathLike = os.path.join(args.results_folder, output_folder, f"{algorithm_name}_{param_value:.5f}.pdf")

            # Calculate the number of clusters
            num_clusters = len(results)

            # Define the legend with dummy handles
            legend = {
                "handles": [
                    Line2D([0], [0], color="black", label=f"Modularidad (Q): {values_0:.4f}"),
                    Line2D([0], [0], color="black", label=f"Significancia Biológica (BSQ): {values_1:.4f}"),
                    Line2D([0], [0], color="black", label=f"Clusters: {num_clusters}")
                ],
                "labels": []  # Labels are embedded directly in the handles
            }

            # Save the clustering visualization
            network_worker.visualize_clusters(
                output_path=output_path,
                clusters=results,
                title="",
                legend=legend,
            )

            # Store the saved image path
            saved_clustering_paths[algorithm_name][param_value] = (output_path, param_name)

            # Log the results
            logger.info(
                f"Algorithm: {algorithm_name}, Parameter: {param_name}={param_value}, "
                f"Configuration: (values_0={values_0}, values_1={values_1})"
            )

    # Separate visualization for "Fast Greedy"
    fastgreedy_algorithm_name = "fastgreedy"
    results: List[List[int]] = Algorithms.fastgreedy_clustering(graph=graph, logger=logger)

    # Calculate modularity and enrichment score
    modularity: float = Metrics.modularity(graph=graph, clusters=results, logger=logger)
    enrichment_score: float = Metrics.functional_enrichment_score(graph=graph, clusters=results, logger=logger)

    # Define the output path for "Fast Greedy"
    fastgreedy_output_path: os.PathLike = os.path.join(args.results_folder, output_folder, f"{fastgreedy_algorithm_name}.pdf")

    # Define the legend for "Fast Greedy"
    num_clusters = len(results)
    fastgreedy_legend = {
        "handles": [
            Line2D([0], [0], color="black", label=f"Modularidad (Q): {modularity:.4f}"),
            Line2D([0], [0], color="black", label=f"Significancia Biológica (BSQ): {enrichment_score:.4f}"),
            Line2D([0], [0], color="black", label=f"Clusters: {num_clusters}")
        ],
        "labels": []  # Labels are embedded directly in the handles
    }

    # Save the clustering visualization for "Fast Greedy"
    network_worker.visualize_clusters(
        output_path=fastgreedy_output_path,
        clusters=results,
        title="",
        legend=fastgreedy_legend,
    )

    # Log the results for "Fast Greedy"
    logger.info(
        f"Algorithm: {fastgreedy_algorithm_name}, "
        f"Modularity: {modularity:.4f}, Enrichment: {enrichment_score:.4f}, Clusters: {num_clusters}"
    )

    saved_clustering_paths[fastgreedy_algorithm_name] = {
        0: (fastgreedy_output_path, "Sin ajuste")
    }

    # Iterate over clustering methods and create one visualization per method
    for algorithm_name, param_data in saved_clustering_paths.items():
        # Extract only the data for the current algorithm
        single_algorithm_data = {algorithm_name: param_data}

        # Call the plotting function for the single algorithm
        plot_saved_clustering_results(single_algorithm_data)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()