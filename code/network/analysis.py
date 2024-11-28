#!/usr/bin/env python3

import argparse
import os
import logging
import utils.misc as utils
import network
import pandas as pd


# Create the logs directory
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok=True)

# Configure logging of this stage of the project
logging.basicConfig(
    filename=os.path.join(log_folder, "network_analysis_logging.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():

    parser = argparse.ArgumentParser(description="Analysis an interaction network")
    parser.add_argument(
        "network",
        type=str,
        help="Path to the file containing the network (e.g., data/genes.tsv)",
    )

    args = parser.parse_args()
    if not os.path.isfile(args.network):
        print(f"Error: The network file '{args.network}' does not exist.")
        logging.error(f"The gene file '{args.network}' does not exist.")
        return

    # Convert network to igraph format
    graph = utils.network_to_igraph_format(args.network)

    analyzer = network.Network(graph)
    # network metrics
    analyzer.calculate_metrics()
    # plot network
    analyzer.visualize_network("../results/network.png")

    metrics = analyzer.metrics
    # falta modularidad que esta en la rama de clustering

    # create a latex table
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df.to_latex(
        buf="../results/networkAnalysisMetrics.tex",
        index=False,
        header=True,
        caption="Network Metrics Summary",
        label="tab:Networlmetrics",
        escape=False,
    )


if __name__ == "__main__":
    main()
