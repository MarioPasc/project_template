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
import argparse
import matplotlib.pyplot as plt
from matplotlib.image import imread

from algorithms import Algorithms
from network import network
from utils import misc

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

def plot_saved_clustering_results(saved_clustering_paths: Dict[str, Dict[float, str]]) -> None:
    """
    Display saved clustering plots in a grid with rows as algorithms and columns as parameter values.

    Args:
        saved_clustering_paths (Dict[str, Dict[float, str]]): Dictionary of clustering image paths.
    """
    num_algorithms = len(saved_clustering_paths)
    max_params_per_algorithm = max(len(params) for params in saved_clustering_paths.values())

    # Create a figure with subplots
    fig, axs = plt.subplots(
        num_algorithms, max_params_per_algorithm,
        figsize=(5 * max_params_per_algorithm, 5 * num_algorithms),
        constrained_layout=True
    )

    # Ensure axs is always a 2D array
    if num_algorithms == 1:
        axs = [axs]
    if max_params_per_algorithm == 1:
        axs = [[ax] for ax in axs]

    # Loop through algorithms and parameter values
    for row_idx, (algorithm_name, param_paths) in enumerate(saved_clustering_paths.items()):
        for col_idx, (param_value, img_path) in enumerate(param_paths.items()):
            # Read and display the saved image
            img = imread(img_path)
            axs[row_idx][col_idx].imshow(img)
            axs[row_idx][col_idx].set_title(f"{algorithm_name} (param: {param_value:.2f})")
            axs[row_idx][col_idx].axis("off")  # Turn off axes for cleaner display

        # Hide unused columns
        for col_idx in range(len(param_paths), max_params_per_algorithm):
            axs[row_idx][col_idx].axis("off")

    plt.show()


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
    os.makedirs(os.path.join(args.results_folder, output_folder), exist_ok=True)

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

            # Save the clustering visualization
            network_worker.visualize_clusters(
                output_path=output_path,
                clusters=results,
                title=f"Cluster {algorithm_name}, Parameter: {param_value}",
                legend=None,
            )

            # Store the saved image path
            saved_clustering_paths[algorithm_name][param_value] = output_path

            # Log the results
            logger.info(
                f"Algorithm: {algorithm_name}, Parameter: {param_name}={param_value}, "
                f"Configuration: (values_0={values_0}, values_1={values_1})"
            )
    plot_saved_clustering_results(saved_clustering_paths=saved_clustering_paths)
if __name__ == "__main__":
    main()