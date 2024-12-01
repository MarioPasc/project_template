#!/usr/bin/env python3

import argparse
import logging
import os

from network import Network
import pandas as pd
from utils import misc as utils

VERBOSE: bool = os.environ.get("VERBOSE", "0") == "1"


logger = utils.setup_logger(
    name="Analysis_PPI_Network",
    log_file=os.path.join("logs/network_analysis_logging.log")
)


def main():

    parser = argparse.ArgumentParser(description="Analysis an interaction network")
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
        "-f",
        "--format",
        type=str,
        default="pdf",
        help="Format to save the imagen (ej. pdf, png, svg...)"
    )

    args = parser.parse_args()
    if not os.path.isfile(args.network):
        logger.error(f"The network file '{args.network}' does not exist.")
        return
    else:
        logger.info(f"Network file '{args.network}' found.")
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)
        logger.info(f"Results folder '{args.results}' created.")
    else:
        logger.info(f"Results folder '{args.results}' already exists.")
    
    # Convert network to igraph format
    try:
        graph = utils.network_to_igraph_format(args.network)
        logger.info("Network conversion successful.")
    except Exception as e:
        logger.error(f"Error converting network to igraph format: {e}")
        return

    network_path_plots: os.PathLike = os.path.join(args.results, 'plots/network')
    os.makedirs(network_path_plots, exist_ok=True)

    if VERBOSE:
        print(f"Starting statistical network analysis. Plotting results in {network_path_plots}.")
        print()

    analyzer = Network(graph, logger)
    # network metrics
    analyzer.calculate_metrics(network_path_plots, args.format)
    # plot network
    analyzer.visualize_network(f"{network_path_plots}/network.{args.format}")

    metrics = analyzer.metrics

    # create a latex table
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df.to_latex(
        buf=f"{args.results}/networkAnalysisMetrics.tex",
        index=False,
        header=True,
        caption="Network Metrics Summary",
        label="tab:Networlmetrics",
        escape=False,
    )

    if VERBOSE:
        print(f"Network successfully analyzed. Check {args.results} folder for results and logs for details.")

if __name__ == "__main__":
    main()
