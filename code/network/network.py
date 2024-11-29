import logging
import math
import statistics
from itertools import combinations
from typing import List

from metrics import Metrics

import igraph
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch


class Network:
    """
    Class to carry out graph analysis using igraph
    """

    def __init__(self, graph: igraph.Graph, logger: logging.Logger)->None:
        """
        Initialize the NetworkAnalysis object

        Args:
            graph (igraph.Graph):  igraph.Graph object
        """

        self.logger: logging.Logger= logger
        if not isinstance(graph, igraph.Graph):
            self.logger.error("ValueError: 'graph' needs to be an igraph.Graph object.")
            raise 

        self.graph: igraph.Graph = graph
        self.metrics: dict = {}
        self.logger.info("Network object initialized successfully with a graph of type: igraph.Graph")

    def find_critical_nodes(self) -> list:
        """
        Find critical nodes in the graph, nodes that make the graph disconnected when are removed.

        Returns:
            list: List of tuples where each tuple contains a set of nodes whose removal would disconnect the graph.
        """
        # not eficient for large vertex_connectivity
        vertex_connectivity = self.graph.vertex_connectivity()
        critical_nodes:list = []

        for nodes in combinations(range(self.graph.vcount()), vertex_connectivity):
            copy_graph = self.graph.copy()
            copy_graph.delete_vertices(nodes)

            if not copy_graph.is_connected():
                critical_nodes.append(nodes)

        return critical_nodes

    def degree(self, result_folder:str, format:str) -> None:
        """
        Analysis of the degree of the nodes. Creates an histograma with the distribution of the degrees
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        try:
            mean, sd  = Metrics.analysis_degree(self.graph, result_folder, format)
        except Exception as e:
            self.logger.error("Exception {e} occur in Metrics.analysis_degree")
        self.metrics["Average degree"] = mean
        self.metrics["Std degree"] = sd
        self.logger.info(f"Degree analysis complete. Average degree: {mean}, Std degree: {sd}")

    def connectivity(self, result_folder:str, format:str)->None:
        """
        Analysis of the connectivity of the network
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        self.metrics["Connected graph"] = self.graph.is_connected()
        self.metrics["Node connectivity"] = self.graph.vertex_connectivity()
        self.metrics["Edge connectivity"] = self.graph.edge_connectivity()

        critical_nodes = self.find_critical_nodes()  # list of tuples

        self.logger.info(f"Connectivity analysis complete")

        cn = set(
            e for tup in critical_nodes for e in tup
        )  # obtain the nodes from the list of tuples

        # highlight in a plot the critical nodes
        self.visualize_network_matplotlib_save(
            output_path=f"{result_folder}/critical_nodes_graph.{format}",
            attributes={
                "vertex_color": [
                    "#FF0000" if n in cn else "#AAAAAA"
                    for n in range(len(self.graph.vs))
                ]
            },
            title="Critical Nodes",
            legend={
                "handles": [Patch(facecolor="red", edgecolor="black")],
                "labels": ["Critical Node"],
            },
        )

    def density(self)->None:
        """
        Compute density and sparsity of the  network
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        density = self.graph.density()
        self.metrics["Density"] = density
        self.metrics["Sparsity"] = 1 - density
        self.logger.info(f"Density: {density}, Sparsity: {1 - density}")

    def closeness_betweenness(self, result_folder:str, format:str)->None:
        """
        Analysis of closennes and betweenness. And visualization of the network based on these metrics.
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        # closseness
        closeness = self.graph.closeness()
        self.metrics["Average Closeness"] = statistics.mean(closeness)
        self.metrics["Std Closeness"] = statistics.stdev(closeness)

        # betweenness
        betweenness = self.graph.betweenness()
        self.metrics["Average betweenness"] = statistics.mean(betweenness)
        self.metrics["Std betweenness"] = statistics.stdev(betweenness)

        self.logger.info(f"Closeness: Average = {self.metrics['Average Closeness']}, Std = {self.metrics['Std Closeness']}")
        self.logger.info(f"Betweenness: Average = {self.metrics['Average betweenness']}, Std = {self.metrics['Std betweenness']}")

        legend = {
                "handles": [
                        Patch(facecolor="#0000FF", edgecolor="black"), 
                        Patch(facecolor="#FF0000", edgecolor="black"), 
                        Patch(facecolor="white", edgecolor="black", label="Node size increases with Betweenness")  
                            ],
                "labels": [
                        "High Closeness (Blue)", 
                        "Low Closeness (Red)",   
                        "Node Size: Betweenness" 
                ]}

        # visualization
        self.visualize_network_matplotlib_save(
            output_path=f"{result_folder}/closeness_betwennes_graph.{format}",
            attributes={
                "vertex_size": [20 + (v / max(betweenness)) * 40 for v in betweenness],
                "vertex_color": [
                    "#%02x%02x%02x" % (int(255 * (1 - c)), 0, int(255 * c))
                    for c in closeness
                ],
            },
            title="Closeness and Betweenness",
            legend=legend
        )
       

    def clustering_coefficients(self, result_folder:str, format:str)->None:
        """
        Compute local transitivity and global transitivity
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """

        # local
        local_transitivity = self.graph.transitivity_local_undirected()
        nan_nodes = [n for n, t in enumerate(local_transitivity) if math.isnan(t)]
        # highlight in the plot nodes with a local_transitivity of NaN (they either don't have neighbourds or only one)
        self.visualize_network_matplotlib_save(
            output_path=f"{result_folder}/transitivity_graph.{format}",
            attributes={
                "vertex_color": [
                    "#FF0000" if n in nan_nodes else "#AAAAAA"
                    for n in range(len(self.graph.vs))
                ]
            },
            title=" Local Transitivity",
            legend={
                "handles": [Patch(facecolor="red", edgecolor="black")],
                "labels": ["Node with NaN Local Transitivity"],
            },
        )

        clean_lt = [
            0 if math.isnan(t) else t for t in local_transitivity
        ]  # drop the NaN values to compute mean and stdev
        self.metrics["Average Local transitivity"] = statistics.mean(clean_lt)
        self.metrics["Std Local transitivity"] = statistics.stdev(clean_lt)

        # global
        self.metrics["Global transitivity"] = self.graph.transitivity_undirected()

        self.logger.info(f"Average Local Transitivity: {self.metrics['Average Local transitivity']}")
        self.logger.info(f"Global Transitivity: {self.metrics['Global transitivity']}")
        self.logger.info(f"Found {len(nan_nodes)} nodes with NaN local transitivity")

    def calculate_metrics(self, result_folder:str, format:str)->None:
        """
        Computes various statistical and structural metrics of the graph, as well as representation using these metrics.
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        self.logger.info("Starting calculation of graph metrics.")
        self.metrics["Number of nodes"] = self.graph.vcount()
        self.metrics["Number of edges"] = self.graph.ecount()
        self.logger.info(f"Num node:{self.graph.vcount()}  Num edges:{self.graph.ecount()}")

        # Degree Distribution
        self.degree(result_folder, format)

        # Connectivity
        self.connectivity(result_folder, format)

        # Density
        self.density()

        # Closeness and betweenness
        self.closeness_betweenness(result_folder, format)

        # Clustering Coeficients
        self.clustering_coefficients(result_folder, format)

        self.metrics["Assortativity"] = self.graph.assortativity_degree()

        self.metrics["Diameter"] = self.graph.diameter()
        self.metrics["Average Path Length"] = self.graph.average_path_length()

    def visualize_network(
        self,
        output_path: str,
        attributes: dict = None,
        default_size: int = 35,
        default_color: str = "red",
    )->None:
        """
        Visualize the graph using igraph.plot, with options to customize the size and color of the nodes.

        Args:
            output_path (str): Path where the generated image will be saved.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization, can include "vertex_size" and "vertex_color".
            default_size (int, optional): Default size of the nodes. Default is 35.
            default_color (str, optional): Default color of the nodes. Default is "red".
        """
        num_nodes = len(self.graph.vs)
        vertex_size = [default_size] * num_nodes
        vertex_color = [default_color] * num_nodes

        if attributes:
            if "vertex_size" in attributes:
                if len(attributes["vertex_size"]) != num_nodes:
                    self.logger(f"ValueError(El tamaño de 'vertex_size' no coincide con el número de nodos ({num_nodes}).)")
                    raise 
                vertex_size = attributes["vertex_size"]
            if "vertex_color" in attributes:
                if len(attributes["vertex_color"]) != num_nodes:
                    self.logger(f"El tamaño de 'vertex_color' no coincide con el número de nodos ({num_nodes}).")
                    raise 
                vertex_color = attributes["vertex_color"]

        igraph.plot(
            self.graph,
            target=output_path,
            vertex_size=vertex_size if vertex_size else default_size,
            vertex_color=vertex_color,
            vertex_label=self.graph.vs["name"],
            vertex_label_size=8,
            edge_width=0.5,
        )
        self.logger.info(f"Visualization saved in {output_path}")

    def visualize_network_matplotlib_save(
        self,
        output_path: str = None,
        attributes: dict = None,
        title: str = None,
        legend: dict = None,
    ) -> None:
        """
        Visualize the graph using igraph and matplotlib and save it in file

        Args:
            out_path (str): Path where the generated image will be saved.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization.
            title (str, optional): Title of the figure.
            legend(dic, optional): Dictionary of legend elements


        Returns:
         ax (matplotlib.Axes): visualization
        """

        fig, ax = plt.subplots(figsize=(8, 8))

        self.visualize_network_matplotlib(ax, attributes, title, legend)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        self.logger.info(f"Visualization saved in {output_path}")

    def visualize_network_matplotlib(
        self, ax: Axes, attributes: dict = None, title: str = None, legend: dict = None
    ) -> Axes:
        """
        Visualize the graph using igraph and matplotlib, with options to customize the size, color of the nodes and edges,
        and add a legend to the plot.

        Args:
            ax (matplotlib.Axes): The axes to plot on.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization.
            title (str, optional): Title of the figure.
            legend(dic, optional): Dictionary of legend elements


        Returns:
         ax (matplotlib.Axes): visualization
        """

        self.visualize_network(ax, attributes, default_size=25)

        if title:
            ax.set_title(title)
        plt.axis("off")

        if legend:
            ax.legend(
                handles=legend["handles"],
                labels=legend["labels"],
                loc="upper right",
                fontsize=10,
            )

        return ax

    def visualize_clusters(
        self,
        output_path: str,
        clusters: List[List[int]],
        title: str = None,
        legend: dict = None,
    ) -> Axes:
        """
        Displays the network with different colors for each cluster.

        Args:
        output_path (str): Name of the file where the network image is save.
        clusters (List[List[int]]): List of lists, where each sublist contains the indexes of the nodes of a cluster.
        title (str, optional): Title of the figure.
        legend(dic, optional): Dictionary of legend elements

        Return:
         ax (matplotlib.Axes): visualization
        """

        num_clusters = len(clusters)
        color_palette = plt.cm.get_cmap("tab20", num_clusters)

        vertex_color = [
            0
        ] * self.graph.vcount()  # create a list of lenght of the number of nodes

        for num, cluster in enumerate(clusters):
            for node in cluster:
                vertex_color[node] = mcolors.to_hex(color_palette(num))

        return self.visualize_network_matplotlib(
            output_path,
            attributes={"vertex_color": vertex_color},
            title=title,
            legend=legend,
        )
