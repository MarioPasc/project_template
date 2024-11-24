#!/usr/bin/env python3
import igraph 
import pandas as pd
import plotly.graph_objects as go
import argparse
import os
import logging
from typing import Tuple
import utils
import statistics
import math
from itertools import combinations

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
            tuple(float, float): Mean degree and standard deviation of the degree distribution.
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
                align="right",  
                bgcolor="white",
                opacity=0.8  
            )

            #fig.show()
            fig.write_html("../results/degree_distribution.html")
            return mean_degree, sd_degree
        except ValueError as v:
            logging.error(f"ValueError: {v}")
            raise


class NetworkAnalysis:
    """
    Class to carry out graph analysis using igraph
    """
    def __init__(self, graph: igraph.Graph):
        """
        Initialize the NetworkAnalysis object

        Args:
            graph (igraph.Graph):  igraph.Graph object
        """

        if not isinstance(graph, igraph.Graph):
            raise ValueError("'graph' needs to be an igraph.Graph object.")

        self.graph=graph
        self.metrics={}


    def find_critical_nodes(self) -> list:
        """
        Find critical nodes in the graph, nodes that make the graph disconnected when are removed.

        Returns:
            list: List of tuples where each tuple contains a set of nodes whose removal would disconnect the graph.
        """
    # not eficient for large vertex_connectivity  
        vertex_connectivity=self.graph.vertex_connectivity() 
        critical_nodes = []  
    
        for nodes in combinations(range(self.graph.vcount()), vertex_connectivity):
            copy_graph = self.graph.copy() 
            copy_graph.delete_vertices(nodes)  

            if not copy_graph.is_connected():
                critical_nodes.append(nodes)
    
        return critical_nodes
    
    def calculate_metrics(self):
        """
        Computes various statistical and structural metrics of the graph, as well as representation using these metrics.
        """
        self.metrics["Number of nodes"]=self.graph.vcount()
        self.metrics["Number of edges"]=self.graph.ecount()
        
        #Degree distribution
        mean, sd=Metrics.analysis_degree(self.graph)
        self.metrics["Average degree"]=mean
        self.metrics["Std degree"]=sd

        #Connectivity
        self.metrics["Connected graph"]= self.graph.is_connected()
        self.metrics["Node connectivity"]=self.graph.vertex_connectivity() 
        self.metrics["Edge connectivity"]=self.graph.edge_connectivity()
        critical_nodes= self.find_critical_nodes() #list of tuples
        cn=set(e for tup in critical_nodes for e in tup)

        self.visualize_network(
            output_path="../results/criticalnodes_graph.png",
            attributes={"vertex_color": ["#FF0000" if n in cn else "#AAAAAA" for n in range(len(self.graph.vs))]}
        )

        density=self.graph.density()
        self.metrics["Density"]=density
        self.metrics["Sparsity"]= 1-density

        closeness=self.graph.closeness()
        self.metrics["Average Closeness"]= statistics.mean(closeness)
        self.metrics["Std Closeness"]= statistics.stdev(closeness)
        #betweenness (centralidad)
        betweenness=self.graph.betweenness()
        self.metrics["Average betweenness"]= statistics.mean(betweenness)
        self.metrics["Std betweenness"]= statistics.stdev(betweenness)
        self.visualize_network(
            output_path="../results/closeness_betwennes_graph.png",
            attributes={"vertex_size": [20 + (v / max(betweenness)) * 40 for v in betweenness], "vertex_color": ["#%02x%02x%02x" % (int(255 * (1 - c)), 0, int(255 * c)) for c in closeness]}
        )

        #Clustering Coeficiente
        #local
        local_transitivity= self.graph.transitivity_local_undirected()
        nan_nodes=[n for n, t in enumerate(local_transitivity) if math.isnan(t)]
        self.visualize_network(
            output_path="../results/transitivity_graph.png",
            attributes={"vertex_color": ["#FF0000" if n in nan_nodes else "#AAAAAA" for n in range(len(self.graph.vs))]}
        )



        clean_lt=[0  if math.isnan(t) else t for t in local_transitivity]
        self.metrics["Average Local transitivity"]= statistics.mean(clean_lt)
        self.metrics["Std Local transitivity"]= statistics.stdev(clean_lt)
            #salen valores NaN -> hay nodos con 1 solo vecino

        #global
        self.metrics["Global transitivity"]=self.graph.transitivity_undirected()

        self.metrics["Assortativity"]=self.graph.assortativity_degree()

        self.metrics["Diameter"] = self.graph.diameter()
        self.metrics["Average Path Length"] = self.graph.average_path_length()


    def visualize_network(self, output_path: str, attributes:dict=None, default_size:int=35, default_color:str="red"):
        """
        Visualize the graph using igraph.plot, with options to customize the size and color of the nodes.

        Args:
            output_path (str): Path where the generated image will be saved.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization, can include "vertex_size" and "vertex_color".
            default_size (int, optional): Default size of the nodes. Default is 35.
            default_color (str, optional): Default color of the nodes. Default is "red".
        """
        num_nodes= len(self.graph.vs)
        vertex_size = [default_size] * num_nodes
        vertex_color = [default_color] * num_nodes

        if attributes:
            if "vertex_size" in attributes:
                if len(attributes["vertex_size"]) != num_nodes:
                    raise ValueError(f"El tamaño de 'vertex_size' no coincide con el número de nodos ({num_nodes}).")
                vertex_size = attributes["vertex_size"]
            if "vertex_color" in attributes:
                if len(attributes["vertex_color"]) != num_nodes:
                    raise ValueError(f"El tamaño de 'vertex_color' no coincide con el número de nodos ({num_nodes}).")
                vertex_color = attributes["vertex_color"]

    
        igraph.plot(
            self.graph,
            target=output_path,
            vertex_size= vertex_size if vertex_size else default_size,
            vertex_color=vertex_color,
            vertex_label=self.graph.vs["name"],
            vertex_label_size=8
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
    graph=utils.network_to_igraph_format(args.network)

    analyzer = NetworkAnalysis(graph)
    # network metrics
    analyzer.calculate_metrics()
    #plot network
    analyzer.visualize_network("../results/network.png")
    

    #TODO find critical

    metrics= analyzer.metrics
    #falta modularidad que esta en la rama de clustering
    
    #create a latex table
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    df.to_latex( buf="../results/networkAnalysisMetrics",index=False, header=True, caption="Network Metrics Summary", label="tab:Networlmetrics", escape=False)



if __name__=="__main__":
    main()