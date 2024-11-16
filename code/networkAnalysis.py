#!/usr/bin/env python3
import igraph as ig
from igraph import Graph
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import logging

# Create the logs directory
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok = True)

# Configure logging of this stage of the project
logging.basicConfig(
    filename=os.path.join(log_folder, 'network_analysis_logging.log'), 
    level=logging.DEBUG,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():

    parser = argparse.ArgumentParser(
        description='Analysis an interaction network'
    )
    parser.add_argument(
        'network', 
        type=str, 
        help='Path to the file containing the network (e.g., data/genes.tsv)'
    )

    args=parser.parse_args()
    if not os.path.isfile(args.network):
        print(f"Error: The network file '{args.network}' does not exist.")
        logging.error(f"The gene file '{args.network}' does not exist.")
        return
    
    # Convert network to igraph format
    network_df= pd.read_csv(args.network, sep="\t", header=0)
    
    graph= Graph.DataFrame(network_df[['preferredName_A', 'preferredName_B']], directed=False, use_vids=False)


    #print(f"Number of nodes {graph.vcount()}")
    #print(f"Number of edges {graph.ecount()}")


    ig.plot(graph, target="../results/network1.svg",vertex_size=35,vertex_label=graph.vs["name"], vertex_label_size=8)


if __name__=="__main__":
    main()