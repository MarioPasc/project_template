#!/usr/bin/env python3

#  - Walktrap: 
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_walktrap. 
#       Se debe poder ajustar el parámetro "steps"
#       Referencia: 
#           PONS, Pascal; LATAPY, Matthieu. 
#           Computing communities in large networks using random walks. 
#           En Computer and Information Sciences-ISCIS 2005: 20th International Symposium, 
#           Istanbul, Turkey, October 26-28, 2005. Proceedings 20. Springer Berlin Heidelberg, 2005. p. 284-293.
#
#   - Leiden:
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_leiden
#       Se debe poder ajustar "resolution". Fijaremos n_iterations a -1 y objective_function a "modularity"
#       Referencia:
#           TRAAG, Vincent A.; WALTMAN, Ludo; VAN ECK, Nees Jan. 
#           From Louvain to Leiden: guaranteeing well-connected communities. 
#           Scientific reports, 2019, vol. 9, no 1, p. 1-12.
#
#   - Louvain:
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_multilevel
#       Se debe poder ajustar "resolution". Fijaremos return_levels = False, en un principio. 
#       Referencia:
#           BLONDEL, Vincent D., et al. 
#           Fast unfolding of communities in large networks. 
#           Journal of statistical mechanics: theory and experiment, 2008, vol. 2008, no 10, p. P10008.

from igraph import Graph, VertexClustering
from typing import List, Optional, Union
import logging

class Algorithms:

    @staticmethod
    def multilevel_clustering(
        graph: Graph,
        logger: logging.Logger,
        weights: Optional[Union[str, List[float]]] = None,
        return_levels: bool = False,
        resolution: float = 1.0,
    ) -> Union[List[List[int]], List[List[List[int]]]]:
        """
        Perform clustering using the Louvain (multilevel) algorithm.

        :param graph: An igraph object representing the graph to be clustered.
        :param logger: A Logger object to log information and errors.
        :param weights: Edge weights, either as a list or a string representing the edge attribute.
        :param return_levels: If True, return all hierarchical levels of clustering.
        :param resolution: Resolution parameter for the algorithm. Higher values detect smaller communities.
        :return: A list of clusters (or a list of hierarchical levels if return_levels=True), 
                 where each cluster is a list of integers representing node indices.
        """
        try:
            # Validate input type
            if not isinstance(graph, Graph):
                raise TypeError("The parameter 'graph' must be an instance of igraph.Graph.")
            
            # Log initial parameters
            logger.info("Iniciando el algoritmo de clustering multilevel.")
            logger.info(f"Parámetros: weights={weights}, return_levels={return_levels}, resolution={resolution}")

            # Perform clustering
            clustering = graph.community_multilevel(
                weights=weights,
                return_levels=return_levels,
                resolution=resolution,
            )

            # Log results
            if return_levels:
                logger.info(f"Clustering completado con múltiples niveles. Niveles detectados: {len(clustering)}")
                # Convert each level to list
                return [Algorithms.convert_clustering_to_list(level, logger) for level in clustering]
            else:
                logger.info(f"Clustering completado. Número de clusters detectados: {len(clustering)}")
                return Algorithms.convert_clustering_to_list(clustering, logger)

        except Exception as e:
            # Log detailed error
            logger.error(f"Error al ejecutar el algoritmo multilevel clustering: {e} "
                         f"con parámetros: weights={weights}, return_levels={return_levels}, resolution={resolution}")
            raise
    
    @staticmethod
    def leiden_clustering(
        graph: Graph,
        logger: logging.Logger,
        weights: Optional[Union[str, List[float]]] = None,
        initial_membership: Optional[List[int]] = None,
        resolution: float = 1.0,
        beta: float = 0.01,
        n_iterations: int = -1,
        objective_function: str = "modularity",
    ) -> List[List[int]]:
        """
        Perform clustering using the Leiden algorithm.

        :param graph: An igraph object representing the graph to be clustered.
        :param logger: A Logger object to log information and errors.
        :param weights: Edge weights, either as a list or a string representing the edge attribute.
        :param initial_membership: Initial membership for nodes, useful for incremental optimization.
        :param resolution: Resolution parameter. Higher values detect smaller communities.
        :param beta: Parameter for randomness. Higher values increase randomness in clustering.
        :param n_iterations: Number of iterations. Using a negative number of iterations will run until a stable iteration is encountered
        :param objective_function: Objective function to optimize ('modularity' or 'CPM').
        :return: A list of clusters, where each cluster is a list of integers representing node indices.
        """
        try:
            # Validate input type
            if not isinstance(graph, Graph):
                raise TypeError("The parameter 'graph' must be an instance of igraph.Graph.")
            
            # Log initial parameters
            logger.info("Iniciando el algoritmo de clustering Leiden.")
            logger.info(f"Parámetros: weights={weights}, initial_membership={initial_membership}, "
                        f"resolution={resolution}, beta={beta}, objective_function={objective_function}")

            # Perform clustering
            clustering = graph.community_leiden(
                weights=weights,
                initial_membership=initial_membership,
                n_iterations = n_iterations,
                resolution_parameter=resolution,
                beta=beta,
                objective_function=objective_function,
            )

            # Log results
            logger.info(f"Clustering completado. Número de clusters detectados: {len(clustering)}")

            # Convert clustering to list
            return Algorithms.convert_clustering_to_list(clustering, logger)

        except Exception as e:
            # Log detailed error
            logger.error(f"Error al ejecutar el algoritmo Leiden clustering: {e} "
                         f"con parámetros: weights={weights}, resolution={resolution}, beta={beta}, "
                         f"objective_function={objective_function}")
            raise

    @staticmethod
    def walktrap_clustering(
        graph: Graph,
        logger: logging.Logger,
        weights: Optional[Union[str, List[float]]] = None,
        steps: int = 4,
    ) -> List[List[int]]:
        """
        Perform clustering using the Walktrap algorithm.

        :param graph: An igraph object representing the graph to be clustered.
        :param logger: A Logger object to log information and errors.
        :param weights: Edge weights, either as a list or a string representing the edge attribute.
        :param steps: The number of steps for the random walk. Higher values lead to larger communities.
        :return: A list of clusters, where each cluster is a list of integers representing node indices.
        """
        try:
            # Validate input type
            if not isinstance(graph, Graph):
                raise TypeError("The parameter 'graph' must be an instance of igraph.Graph.")
            
            # Log initial parameters
            logger.info("Iniciando el algoritmo de clustering Walktrap.")
            logger.info(f"Parámetros: weights={weights}, steps={steps}")

            # Perform clustering
            dendrogram = graph.community_walktrap(weights=weights, steps=steps)
            clustering = dendrogram.as_clustering()

            # Log results
            logger.info(f"Clustering completado. Número de clusters detectados: {len(clustering)}")

            # Convert clustering to list
            return Algorithms.convert_clustering_to_list(clustering, logger)

        except Exception as e:
            # Log detailed error
            logger.error(f"Error al ejecutar el algoritmo Walktrap clustering: {e} "
                         f"con parámetros: weights={weights}, steps={steps}")
            raise

    @staticmethod
    def fastgreedy_clustering(
        graph: Graph,
        logger: logging.Logger
    ) -> List[List[int]]:
        """
        Perform clustering using the FastGreedy algorithm.

        :param graph: An igraph object representing the graph to be clustered.
        :param logger: A Logger object to log information and errors.
        :return: A list of clusters, where each cluster is a list of integers representing node indices.
        """
        try:
            # Validate input type
            if not isinstance(graph, Graph):
                raise TypeError("The parameter 'graph' must be an instance of igraph.Graph.")
            
            # Log the start of clustering
            logger.info("Iniciando el algoritmo de clustering FastGreedy.")

            # Perform clustering
            dendrogram = graph.community_fastgreedy()
            clustering = dendrogram.as_clustering()

            # Log the results
            logger.info(f"Clustering completado. Número de clusters detectados: {len(clustering)}")

            # Convert clustering to list
            return Algorithms.convert_clustering_to_list(clustering, logger)

        except Exception as e:
            # Log detailed error
            logger.error(f"Error al ejecutar el algoritmo FastGreedy clustering: {e}")
            raise

    @staticmethod
    def convert_clustering_to_list(clustering: VertexClustering, logger: logging.Logger) -> List[List[int]]:
        """
        Converts a VertexClustering object into a list of lists format.

        :param clustering: A VertexClustering object from igraph.
        :param logger: A Logger object to log information and errors.
        :return: A list of lists where each sublist represents the vertices in a cluster.
        """
        try:
            # Extract clusters as a list of lists
            cluster_list = [list(cluster) for cluster in clustering]
            return cluster_list
        except Exception as e:
            logger.error(f"Error converting VertexClustering to list: {e}")