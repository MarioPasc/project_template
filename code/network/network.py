#!/usr/bin/env python3

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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scienceplots

# Style
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "300"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


import os

VERBOSE: bool = os.environ.get("VERBOSE", "0") == "1"
import time


class Network:
    """
    Class to carry out graph analysis using igraph
    """

    def __init__(self, graph: igraph.Graph, logger: logging.Logger) -> None:
        """
        Initialize the NetworkAnalysis object

        Args:
            graph (igraph.Graph):  igraph.Graph object
        """

        self.logger: logging.Logger = logger
        if not isinstance(graph, igraph.Graph):
            self.logger.error("ValueError: 'graph' needs to be an igraph.Graph object.")
            raise

        self.graph: igraph.Graph = graph
        self.metrics: dict = {}
        self.logger.info(
            "Network object initialized successfully with a graph of type: igraph.Graph"
        )

    def find_critical_nodes(self) -> list:
        """
        Find critical nodes in the graph, nodes that make the graph disconnected when are removed.

        Returns:
            list: List of tuples where each tuple contains a set of nodes whose removal would disconnect the graph.
        """
        # not eficient for large vertex_connectivity
        vertex_connectivity = self.graph.vertex_connectivity()
        critical_nodes: list = []

        for nodes in combinations(range(self.graph.vcount()), vertex_connectivity):
            copy_graph = self.graph.copy()
            copy_graph.delete_vertices(nodes)

            if not copy_graph.is_connected():
                critical_nodes.append(nodes)

        return critical_nodes

    def degree(self, result_folder: str, format: str) -> None:
        """
        Analysis of the degree of the nodes. Creates an histograma with the distribution of the degrees
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        try:
            mean, sd = Metrics.analysis_degree(self.graph, result_folder, format)
        except Exception as e:
            self.logger.error("Exception {e} occur in Metrics.analysis_degree")
        self.metrics["Average degree"] = mean
        self.metrics["Std degree"] = sd
        self.logger.info(
            f"Degree analysis complete. Average degree: {mean}, Std degree: {sd}"
        )

    def connectivity(self, result_folder: str, format: str) -> None:
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
            title="Nodos críticos (desconectan el grafo si son eliminados)",
            legend={
                "handles": [Patch(facecolor="red", edgecolor="red")],
                "labels": ["Nodos críticos"],
            },
        )

    def density(self) -> None:
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

    def closeness_betweenness(self, result_folder: str, format: str) -> None:
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

        self.logger.info(
            f"Closeness: Average = {self.metrics['Average Closeness']}, Std = {self.metrics['Std Closeness']}"
        )
        self.logger.info(
            f"Betweenness: Average = {self.metrics['Average betweenness']}, Std = {self.metrics['Std betweenness']}"
        )

        legend = {
            "handles": [
                Patch(
                    facecolor="black",
                    edgecolor="black",
                    label="Tamaño de los nodos incrementa con la centralidad",
                ),
                Patch(
                    facecolor="red",
                    edgecolor="red",
                    label="Color aumenta con la cercanía",
                ),
            ],
            "labels": ["Tamaño nodos: Betweenness", "Color nodos: Closeness"],
        }

        # visualization
        fig, ax = plt.subplots(figsize=(20, 10))

        # normalize closseness values to represent them as a colour scale
        norm = mcolors.Normalize(vmin=min(closeness), vmax=max(closeness))
        cmap = cm.YlOrRd

        self.visualize_network_matplotlib(
            ax,
            attributes={
                "vertex_size": [20 + (v / max(betweenness)) * 40 for v in betweenness],
                "vertex_color": [cmap(norm(c)) for c in closeness],
            },
            title=" Cercanía (closeness) y Centralidad (betweenness)",
            legend=legend,
        )

        # add color bar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Closeness", fontsize=10)

        # save graph
        output_path = f"{result_folder}/closeness_betwennes_graph.{format}"
        plt.savefig(output_path, format=format, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def clustering_coefficients(self, result_folder: str, format: str) -> None:
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
            title="Transitividad Local",
            legend={
                "handles": [Patch(facecolor="red", edgecolor="red")],
                "labels": ["Nodo con transitividad local NaN"],
            },
        )

        clean_lt = [
            0 if math.isnan(t) else t for t in local_transitivity
        ]  # drop the NaN values to compute mean and stdev
        self.metrics["Average Local transitivity"] = statistics.mean(clean_lt)
        self.metrics["Std Local transitivity"] = statistics.stdev(clean_lt)

        # global
        self.metrics["Global transitivity"] = self.graph.transitivity_undirected()

        self.logger.info(
            f"Average Local Transitivity: {self.metrics['Average Local transitivity']}"
        )
        self.logger.info(f"Global Transitivity: {self.metrics['Global transitivity']}")
        self.logger.info(f"Found {len(nan_nodes)} nodes with NaN local transitivity")

    def calculate_metrics(self, result_folder: str, format: str) -> None:
        """
        Computes various statistical and structural metrics of the graph, as well as representation using these metrics.
        Args:
            result_folder(str): Path to the directory where the results will be saved.
            format (str): Format of saved images.
        """
        self.logger.info("Starting calculation of graph metrics.")
        self.metrics["Number of nodes"] = self.graph.vcount()
        self.metrics["Number of edges"] = self.graph.ecount()
        self.logger.info(
            f"Num node:{self.graph.vcount()}  Num edges:{self.graph.ecount()}"
        )

        # Add the time.sleep to not saturate the terminal output

        if VERBOSE:
            print("Network Stats: Computing degree distribution ...")
        # Degree Distribution
        self.degree(result_folder, format)
        if VERBOSE:
            time.sleep(0.5)

        if VERBOSE:
            print("Network Stats: Computing connectivity ...")
        # Connectivity
        self.connectivity(result_folder, format)
        if VERBOSE:
            time.sleep(0.5)

        if VERBOSE:
            print("Network Stats: Computing network density ...")
        # Density
        self.density()
        if VERBOSE:
            time.sleep(0.5)

        if VERBOSE:
            print("Network Stats: Computing closeness and betweenness ...")
        # Closeness and betweenness
        self.closeness_betweenness(result_folder, format)
        if VERBOSE:
            time.sleep(0.5)

        if VERBOSE:
            print("Network Stats: Computing clustering coefficients ...")
        # Clustering Coeficients
        self.clustering_coefficients(result_folder, format)
        if VERBOSE:
            time.sleep(0.5)
        if VERBOSE:
            print()

        self.metrics["Assortativity"] = self.graph.assortativity_degree()

        self.metrics["Diameter"] = self.graph.diameter()
        self.metrics["Average Path Length"] = self.graph.average_path_length()

    def visualize_network(
        self,
        output_path: str,
        attributes: dict = None,
        default_size: int = 35,
        default_color: str = "red",
        layout: str = "auto",
    ) -> None:
        """
        Visualize the graph using igraph.plot, with options to customize the visualization using a flexible attributes dictionary.

        Args:
            output_path (str): Path where the generated image will be saved.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization, such as "vertex_size", "vertex_color", "edge_width", etc.
            default_size (int, optional): Default size of the nodes. Default is 80.
            default_color (str, optional): Default color of the nodes. Default is "red".
            layout (str, optional): Type of layout of the graph (default: auto)
        """
        num_nodes = len(self.graph.vs)

        # Initialize default values
        plot_attributes = {
            "vertex_size": [default_size] * num_nodes,
            "vertex_color": [default_color] * num_nodes,
            "vertex_label": self.graph.vs["name"],
            "vertex_label_size": 8,
            "edge_width": 0.5,
            "margin": 50,
            "vertex_frame_width": 0 
        }

        # Update default values with attributes passed in the dictionary
        if attributes:
            for key, value in attributes.items():
                if (
                    isinstance(value, list)
                    and len(value) != num_nodes
                    and key.startswith("vertex_")
                ):
                    self.logger.error(
                        f"The size of '{key}' does not match the number of nodes ({num_nodes})."
                    )
                    raise ValueError(
                        f"The size of '{key}' does not match the number of nodes ({num_nodes})."
                    )
                plot_attributes[key] = value

        # try if the layout is valid
        try:
            if layout != "auto" and layout != None:
                graph_layout = self.graph.layout(layout)
            else:
                graph_layout = self.graph.layout_auto()
        except ValueError:
            self.logger.error(f"Invalid layout '{layout}' specified.")
            raise ValueError(f"Invalid layout '{layout}' specified.")
        # NOTE: drl, kamada_kawai, fruchterman_reingold no van mal

        # Plot the graph using igraph
        igraph.plot(
            self.graph, target=output_path, layout=graph_layout, **plot_attributes
        )
        self.logger.info(f"Visualization saved in {output_path}")

    def visualize_network_matplotlib_save(
        self,
        output_path: str = None,
        attributes: dict = None,
        title: str = None,
        legend: dict = None,
        layout: str = None,
    ) -> None:
        """
        Visualize the graph using igraph and matplotlib and save it in file

        Args:
            out_path (str): Path where the generated image will be saved.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization.
            title (str, optional): Title of the figure.
            legend(dic, optional): Dictionary of legend elements
            layout (str, optionl): Network layout


        Returns:
         ax (matplotlib.Axes): visualization
        """

        fig, ax = plt.subplots(figsize=(8, 8))

        self.visualize_network_matplotlib(ax, attributes, title, legend, layout)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        self.logger.info(f"Visualization saved in {output_path}")

    def visualize_network_matplotlib(
        self,
        ax: Axes,
        attributes: dict = None,
        title: str = None,
        legend: dict = None,
        layout: str = None,
    ) -> plt.Axes:
        """
        Visualize the graph using igraph and matplotlib, with options to customize the size, color of the nodes and edges,
        and add a legend to the plot.

        Args:
            ax (matplotlib.Axes): The axes to plot on.
            attributes (dict, optional): Dictionary of additional attributes to customize the visualization.
            title (str, optional): Title of the figure.
            legend(dic, optional): Dictionary of legend elements
            layout (str, optionl): Network layout


        Returns:
         ax (matplotlib.Axes): visualization
        """

        self.visualize_network(ax, attributes, default_size=25, layout=layout)

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
        attributes: dict = None,
        layout: str = None,
    ) -> plt.Axes:
        """
        Displays the network with different colors for each cluster. The color of each cluster
        is determined by the cluster's representative node (lowest ID node in the cluster),
        ensuring consistent cluster coloring across different clustering methods.

        Args:
            output_path (str): Path where the network image will be saved.
            clusters (List[List[int]]): List of lists, where each sublist contains the indexes of the nodes of a cluster.
            title (str, optional): Title of the figure.
            legend (dict, optional): Dictionary of legend elements.
            attributes (dict, optional): Attributes dictionary for visual customization.
            layout (str, optional): Network layout (drl, kamada_kawai, fruchterman_reingold )

        Returns:
            plt.Axes: The matplotlib Axes object for the visualization.
        """
        # Determine the representative nodes (lowest ID in each cluster)
        representative_nodes = [min(cluster) for cluster in clusters]

        # Get a sorted list of unique representative nodes to map them consistently to colors
        unique_representatives = sorted(set(representative_nodes))

        # Choose a fixed colormap. "tab10" provides up to 10 distinct colors.
        # If you have more than 10 clusters, consider "tab20" or another larger palette.
        color_map = plt.cm.get_cmap("tab10", len(unique_representatives))

        # Create a dictionary mapping representative_node -> color
        rep_color_dict = {
            rep_node: mcolors.to_hex(color_map(i))
            for i, rep_node in enumerate(unique_representatives)
        }

        # Create a vertex_color list based on cluster representative colors
        vertex_color = [None] * self.graph.vcount()
        for cluster in clusters:
            rep_node = min(cluster)
            cluster_color = rep_color_dict[rep_node]
            for node in cluster:
                vertex_color[node] = cluster_color

        # If attributes dictionary is provided, append vertex_color
        if attributes is None:
            attributes = {}
        attributes["vertex_color"] = vertex_color

        # Create a matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the network on the Axes
        ax = self.visualize_network_matplotlib(
            ax=ax,
            attributes=attributes,  # Pass the updated attributes
            title=title,
            legend=None,  # Legend handled below
            layout=layout,
        )

        # Add legend if provided
        if legend:
            ax.legend(
                handles=legend["handles"],
                loc="lower center",  # Place the legend at the top center relative to the anchor
                bbox_to_anchor=(0.5, -0.05),  # Shift the legend box below the plot
                fontsize=10,
                frameon=False,
            )

        # Save the plot if output_path is provided
        if output_path:
            fig.savefig(output_path, format="png", bbox_inches="tight")

        return ax
