#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd
from igraph import Graph
from functional_analysis import FunctionalAnalysis
from visualization import FunctionalVisualization
from utils import misc as utils
from typing_extensions import LiteralString

# Global flag for verbose logging
VERBOSE: bool = os.environ.get("VERBOSE", "0") == "1"

# Set up a logger for the script
logger = utils.setup_logger(
    name="Functional_Analysis_Visualization",
    log_file=os.path.join("logs/functional_analysis_logging.log"),
)


def functional_analysis_pipeline(
    graph: Graph,
    algorithm: str,
    results_folder: str,
    clustering_json: str,
    adjusted_p_value_threshold: float,
    combined_score_threshold: float,
    overlap_percentage_threshold: float,
) -> LiteralString:
    """
    Executes the functional analysis pipeline:
    1. Performs functional enrichment analysis based on a clustering algorithm.
    2. Filters the results using statistical thresholds.

    :param graph: The igraph object representing the input network.
    :param algorithm: The clustering algorithm used for analysis.
    :param results_folder: Folder to save analysis results.
    :param clustering_json: Path to the clustering JSON data.
    :param adjusted_p_value_threshold: Threshold for filtering results by adjusted p-value.
    :param combined_score_threshold: Minimum combined score for filtering results.
    :param overlap_percentage_threshold: Minimum overlap percentage for filtering results.
    :return: Path to the filtered results CSV file.
    """
    # Step 1: Perform functional analysis
    functional_analysis_results = os.path.join(
        results_folder, f"functional_analysis_{algorithm}.csv"
    )
    try:
        # Initialize FunctionalAnalysis and load clustering data
        fa = FunctionalAnalysis(graph)
        clustering_data = utils.load_json(clustering_json)

        # Perform the functional analysis
        fa.perform_analysis(
            clustering_data=clustering_data,
            output_file=functional_analysis_results,
            algorithm=algorithm,
        )
        logger.info(
            f"Functional analysis results saved to {functional_analysis_results}."
        )
    except Exception as e:
        logger.error(f"Error during functional analysis: {e}")
        return

    # Step 2: Filter results by thresholds
    filtered_results = os.path.join(results_folder, f"filtered_results_{algorithm}.csv")
    try:
        fa.filter_results(
            input_file=functional_analysis_results,
            output_file=filtered_results,
            p_value_threshold=adjusted_p_value_threshold,
            combined_score_min=combined_score_threshold,
            overlap_percentage_min=overlap_percentage_threshold,
        )
        logger.info(f"Filtered results saved to {filtered_results}.")
        return filtered_results
    except Exception as e:
        logger.error(f"Error during result filtering: {e}")
        return


def main():
    """
    Main function to execute the functional analysis and visualization pipeline.
    """
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(
        description="Perform functional analysis and visualize results."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Path to the input network file (e.g., data/genes.tsv)",
    )
    parser.add_argument(
        "results", type=str, help="Directory to save the analysis results."
    )
    parser.add_argument(
        "clustering_json",
        type=str,
        help="Path to the JSON file containing clustering data.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="pdf",
        help="Image format to save the plots (e.g., pdf, png, svg). Default is 'pdf'.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        help="Clustering algorithm for analysis (optional).",
    )
    parser.add_argument(
        "-p",
        "--adjusted_p_value_threshold",
        type=float,
        default=0.05,
        help="Threshold for adjusted p-value filtering (default: 0.05).",
    )
    parser.add_argument(
        "-c",
        "--combined_score_threshold",
        type=float,
        default=2000,
        help="Minimum combined score for filtering results (default: 2000).",
    )
    parser.add_argument(
        "-o",
        "--overlap_percentage_threshold",
        type=float,
        default=0.1,
        help="Minimum overlap percentage for filtering results (default: 0.1).",
    )

    args = parser.parse_args()

    # Validate input files and directories
    if not os.path.isfile(args.network):
        logger.error(f"The network file '{args.network}' does not exist.")
        return
    if not os.path.isfile(args.clustering_json):
        logger.error(
            f"The clustering JSON file '{args.clustering_json}' does not exist."
        )
        return
    if not os.path.exists(args.results):
        os.makedirs(args.results)
        logger.info(f"Created results folder '{args.results}'.")

    # Convert the network file to an igraph object
    try:
        graph = utils.network_to_igraph_format(args.network)
        logger.info("Network successfully converted to igraph format.")
    except Exception as e:
        logger.error(f"Error converting network file to igraph format: {e}")
        return

    # Run the functional analysis pipeline
    filtered_results = functional_analysis_pipeline(
        graph=graph,
        algorithm=args.algorithm,
        results_folder=args.results,
        clustering_json=args.clustering_json,
        adjusted_p_value_threshold=args.adjusted_p_value_threshold,
        overlap_percentage_threshold=args.overlap_percentage_threshold,
        combined_score_threshold=args.combined_score_threshold,
    )

    # Visualization
    try:
        filtered_data = pd.read_csv(filtered_results)
        prepared_data, gene_sets = (
            FunctionalVisualization.prepare_data_for_visualization_from_df(
                filtered_data
            )
        )

        plots_path = os.path.join(args.results, "plots/functional_analysis")
        os.makedirs(plots_path, exist_ok=True)

        # Generate plots
        FunctionalVisualization.dot_plot(
            prepared_data, os.path.join(plots_path, f"dot_plot.{args.format}")
        )
        FunctionalVisualization.bar_plot(
            prepared_data, os.path.join(plots_path, f"bar_plot.{args.format}")
        )
        FunctionalVisualization.cnet_plot_igraph(
            prepared_data,
            gene_sets,
            os.path.join(plots_path, f"cnet_plot.{args.format}"),
        )

        # Venn diagram
        # Our best results were leiden max enrichment, so we perform the analysis
        # for max modularity
        filtered_results_max_modularity = functional_analysis_pipeline(
            graph=graph,
            algorithm="leiden_max_modularity",
            results_folder=args.results,
            clustering_json=args.clustering_json,
            adjusted_p_value_threshold=args.adjusted_p_value_threshold,
            overlap_percentage_threshold=args.overlap_percentage_threshold,
            combined_score_threshold=args.combined_score_threshold,
        )

        # Visualization
        venn_file = os.path.join(plots_path, f"venn_plot.{args.format}")
        FunctionalVisualization.venn_diagram(
            file_enrichment=filtered_results,
            file_modularity=filtered_results_max_modularity,
            output_file=venn_file,
        )

        logger.info("Visualization completed successfully.")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return

    if VERBOSE:
        print(f"Pipeline completed. Results are saved in '{args.results}'.")


if __name__ == "__main__":
    main()
