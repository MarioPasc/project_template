#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd
from functional_analysis import FunctionalAnalysis
from visualization import FunctionalVisualization
from utils import misc as utils

VERBOSE: bool = os.environ.get("VERBOSE", "0") == "1"

logger = utils.setup_logger(
    name="Functional_Analysis_Visualization",
    log_file=os.path.join("logs/functional_analysis_logging.log"),
)


def main():
    parser = argparse.ArgumentParser(
        description="Perform functional analysis and visualize results."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Path to the file containing the network (e.g., data/genes.tsv)",
    )
    parser.add_argument(
        "results",
        type=str,
        help="Folder where the results will be saved",
    )
    parser.add_argument(
        "clustering_json",
        type=str,
        help="Path to the JSON file containing clustering data",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="pdf",
        help="Format to save the images (e.g., pdf, png, svg...)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        help="Specify the clustering algorithm for analysis (optional).",
    )
    parser.add_argument(
        "-p",
        "--p_value_threshold",
        type=float,
        default=0.05,
        help="Threshold for filtering results by p-value (default: 0.05).",
    )

    args = parser.parse_args()

    # Validar archivos y carpetas
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
        logger.info(f"Results folder '{args.results}' created.")

    # Convertir la red al formato igraph
    try:
        graph = utils.network_to_igraph_format(args.network)
        logger.info("Network conversion successful.")
    except Exception as e:
        logger.error(f"Error converting network to igraph format: {e}")
        return

    # Ejecutar análisis funcional
    functional_analysis_results = os.path.join(args.results, "functional_analysis.csv")
    try:
        fa = FunctionalAnalysis(graph)
        clustering_data = utils.load_json(
            args.clustering_json
        )  # Supongamos que esta función carga el JSON
        fa.perform_analysis(
            clustering_data=clustering_data,
            output_file=functional_analysis_results,
            algorithm=args.algorithm,
        )
        logger.info(
            f"Functional analysis completed. Results saved to {functional_analysis_results}."
        )
    except Exception as e:
        logger.error(f"Error during functional analysis: {e}")
        return

    # Filtrar resultados por p-valor
    filtered_results = os.path.join(args.results, "filtered_results.csv")
    try:
        fa.filter_results(
            input_file=functional_analysis_results,
            output_file=filtered_results,
            p_value_threshold=args.p_value_threshold,
        )
        logger.info(f"Filtered results saved to {filtered_results}.")
    except Exception as e:
        logger.error(f"Error during result filtering: {e}")
        return

    # Visualizar resultados
    try:
        filtered_data = pd.read_csv(filtered_results)
        prepared_data, gene_sets = (
            FunctionalVisualization.prepare_data_for_visualization_from_df(
                filtered_data
            )
        )

        plots_path: str = "plots/functional_analysis"
        os.makedirs(os.path.join(args.results, plots_path), exist_ok=True)

        # Dot Plot
        dot_plot_file = os.path.join(
            args.results, os.path.join(plots_path, f"dot_plot.{args.format}")
        )
        FunctionalVisualization.dot_plot(prepared_data, dot_plot_file)

        # Bar Plot
        bar_plot_file = os.path.join(
            args.results, os.path.join(plots_path, f"bar_plot.{args.format}")
        )
        FunctionalVisualization.bar_plot(prepared_data, bar_plot_file)

        # Cnet Plot
        cnet_plot_file = os.path.join(
            args.results, os.path.join(plots_path, f"cnet_plot.{args.format}")
        )
        FunctionalVisualization.cnet_plot(prepared_data, gene_sets, cnet_plot_file)

        logger.info("Visualization completed.")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return

    if VERBOSE:
        print(
            f"Functional analysis and visualization successfully completed. Results saved in '{args.results}'."
        )


if __name__ == "__main__":
    main()
