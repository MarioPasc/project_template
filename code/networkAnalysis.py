#!/usr/bin/env python3
import igraph 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse
import os
import logging
from typing import Tuple
import utils

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

class Metrics:
    @staticmethod
    def analysis_degree(graph: igraph.Graph)-> Tuple[float, float]:
        """
        Analyzes the degree distribution of a graph and saves a histogram in HTML format.

        Args:
            graph (igraph.Graph): The graph to analyze.

        Returns:
            tuple: Mean degree and standard deviation of the degree distribution.
        """
        try:
            if not isinstance(graph, igraph.Graph):
                raise ValueError("The provided graph is not an igraph.Graph object.")
            
            # degree of each node
            #degree_dict = dict(zip(graph.vs["name"], graph.degree()))

            #obtain the degree distribution of the graph
            degree_distribution= graph.degree_distribution()

            mean_degree=degree_distribution.mean
            sd_degree=degree_distribution.sd

            #histogram of the degree distribution using plotly
            bins = list(degree_distribution.bins()) # list of tuples (left_limit, right_limit, count)

            x = [bin[0] for bin in bins]  # left limit
            y = [bin[2] for bin in bins]  # counts

            fig = go.Figure(data=[go.Bar(x=x, y=y)])

            fig.update_layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Frecuency",
            bargap=0.2
            )

            #to include in the histogram the mean and std
            fig.add_annotation(x=34,  y=4,  
                text=f'Mean: {mean_degree:.2f}<br>Standar Deviation: {sd_degree:.2f}',  
                showarrow=False,  
                font=dict(size=14, color="black"),
                align="right",  # Alineación del texto
                bgcolor="white",  # Fondo blanco para la anotación
                opacity=0.8  # Opacidad de la anotación
            )

            #fig.show()
            fig.write_html("../results/degree_distribution.html")
            return mean_degree, sd_degree
        except ValueError as v:
            logging.error(f"ValueError: {v}")
            raise





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
    graph=utils.network_to_igraph_format(args.network)

    #plot network
    igraph.plot(graph, target="../results/network1.svg",vertex_size=35,vertex_label=graph.vs["name"], vertex_label_size=8)

    metrics={}

    metrics["Number of nodes"]=graph.vcount()
    metrics["Number of edges"]=graph.ecount()
    
    #Degree distribution
    mean, sd=Metrics.analysis_degree(graph)
    metrics["Average degree"]=mean
    metrics["Std degree"]=sd

    #Connectivity
    metrics["Connected graph"]= graph.is_connected()
    metrics["Node connectivity"]=graph.vertex_connectivity()
    metrics["Edge connectivity"]=graph.edge_connectivity()

    metrics["Density"]=graph.density()
    closeness=graph.closeness()
    metrics["Average Closeness"]= sum(closeness)/len(closeness)

    #create a latex table
    #df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    #df.to_latex( buf="../results/networkAnalysisMetrics",index=False, header=True, caption="Network Metrics Summary", label="tab:Networlmetrics", escape=False)



if __name__=="__main__":
    main()