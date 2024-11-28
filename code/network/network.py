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

    def __init__(self, graph: igraph.Graph):
        """
        Initialize the NetworkAnalysis object

        Args:
            graph (igraph.Graph):  igraph.Graph object
        """

        if not isinstance(graph, igraph.Graph):
            raise ValueError("'graph' needs to be an igraph.Graph object.")

        self.graph = graph
        self.metrics = {}

    def find_critical_nodes(self) -> list:
        """
        Find critical nodes in the graph, nodes that make the graph disconnected when are removed.

        Returns:
            list: List of tuples where each tuple contains a set of nodes whose removal would disconnect the graph.
        """
        # not eficient for large vertex_connectivity
        vertex_connectivity = self.graph.vertex_connectivity()
        critical_nodes = []

        for nodes in combinations(range(self.graph.vcount()), vertex_connectivity):
            copy_graph = self.graph.copy()
            copy_graph.delete_vertices(nodes)

            if not copy_graph.is_connected():
                critical_nodes.append(nodes)

        return critical_nodes

    def degree(self):
        """
        Analysis of the degree of the nodes
        """
        mean, sd = Metrics.analysis_degree(self.graph)
        self.metrics["Average degree"] = mean
        self.metrics["Std degree"] = sd

    def connectivity(self):
        """
        Analysis of the connectivity of the network
        """
        self.metrics["Connected graph"] = self.graph.is_connected()
        self.metrics["Node connectivity"] = self.graph.vertex_connectivity()
        self.metrics["Edge connectivity"] = self.graph.edge_connectivity()

        critical_nodes = self.find_critical_nodes()  # list of tuples

        cn = set(
            e for tup in critical_nodes for e in tup
        )  # obtain the nodes from the list of tuples

        # highlight in a plot the critical nodes
        self.visualize_network_matplotlib_save(
            output_path="../results/critical_nodes_graph.png",
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

    def density(self):
        """
        Compute density and sparsity of the  network
        """
        density = self.graph.density()
        self.metrics["Density"] = density
        self.metrics["Sparsity"] = 1 - density

    def closeness_betweenness(self):
        """
        Analysis of closennes and betweenness. And visualization of the network based on these metrics.
        """
        # closseness
        closeness = self.graph.closeness()
        self.metrics["Average Closeness"] = statistics.mean(closeness)
        self.metrics["Std Closeness"] = statistics.stdev(closeness)

        # betweenness
        betweenness = self.graph.betweenness()
        self.metrics["Average betweenness"] = statistics.mean(betweenness)
        self.metrics["Std betweenness"] = statistics.stdev(betweenness)

        # visualization
        self.visualize_network(
            output_path="../results/closeness_betwennes_graph.png",
            attributes={
                "vertex_size": [20 + (v / max(betweenness)) * 40 for v in betweenness],
                "vertex_color": [
                    "#%02x%02x%02x" % (int(255 * (1 - c)), 0, int(255 * c))
                    for c in closeness
                ],
            },
        )

    def clustering_coefficients(self):
        """
        Compute local transitivity and global transitivity
        """

        # local
        local_transitivity = self.graph.transitivity_local_undirected()
        nan_nodes = [n for n, t in enumerate(local_transitivity) if math.isnan(t)]
        # highlight in the plot nodes with a local_transitivity of NaN (they either don't have neighbourds or only one)
        self.visualize_network_matplotlib_save(
            output_path="../results/transitivity_graph.png",
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

    def calculate_metrics(self):
        """
        Computes various statistical and structural metrics of the graph, as well as representation using these metrics.
        """
        self.metrics["Number of nodes"] = self.graph.vcount()
        self.metrics["Number of edges"] = self.graph.ecount()

        # Degree Distribution
        self.degree()

        # Connectivity
        self.connectivity()

        # Density
        self.density()

        # Closeness and betweenness
        self.closeness_betweenness()

        # Clustering Coeficients
        self.clustering_coefficients()

        self.metrics["Assortativity"] = self.graph.assortativity_degree()

        self.metrics["Diameter"] = self.graph.diameter()
        self.metrics["Average Path Length"] = self.graph.average_path_length()

    def visualize_network(
        self,
        output_path: str,
        attributes: dict = None,
        default_size: int = 35,
        default_color: str = "red",
    ):
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
                    raise ValueError(
                        f"El tamaño de 'vertex_size' no coincide con el número de nodos ({num_nodes})."
                    )
                vertex_size = attributes["vertex_size"]
            if "vertex_color" in attributes:
                if len(attributes["vertex_color"]) != num_nodes:
                    raise ValueError(
                        f"El tamaño de 'vertex_color' no coincide con el número de nodos ({num_nodes})."
                    )
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
            plt.savefig(output_path, format="png", bbox_inches="tight")

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

        self.visualize_network(ax, attributes)

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
