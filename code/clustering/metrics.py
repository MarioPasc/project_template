#!/usr/bin/env python3

from igraph import Graph
import logging
import numpy as np

class Metrics:
    @staticmethod
    def modularity(graph, clusters, log_file=None):
        """
        Calcula la modularidad (Q) usando igraph y registra eventos importantes en un archivo de registro.

        :param graph: Red como un objeto igraph.Graph.
        :param clusters: Lista de listas donde cada sublista representa un cluster con nodos.
        :param log_file: Ruta a un archivo de registro donde se escriben eventos importantes.
        :return: Un valor flotante representando la modularidad o None si ocurre un error.
        """
        # Configurar logging
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(level=logging.CRITICAL)  # Si no hay log_file, no registra nada.

        try:
            # Validar que el grafo sea un objeto de tipo igraph.Graph
            if not isinstance(graph, Graph):
                raise TypeError("El parámetro 'graph' debe ser un objeto de tipo igraph.Graph.")

            # Validar que 'clusters' sea una lista de listas
            if not isinstance(clusters, list) or not all(isinstance(c, list) for c in clusters):
                raise TypeError("El parámetro 'clusters' debe ser una lista de listas.")

            # Validar que todos los nodos en clusters estén en el rango de nodos del grafo
            num_nodes = graph.vcount()
            if any(node < 0 or node >= num_nodes for cluster in clusters for node in cluster):
                raise ValueError(f"Los nodos en 'clusters' deben estar en el rango 0 a {num_nodes - 1}.")

            # Convertir los clústeres a una lista de asignaciones (membership)
            membership = [None] * num_nodes
            for cluster_index, cluster in enumerate(clusters):
                for node in cluster:
                    membership[node] = cluster_index

            # Validar que todos los nodos tengan una asignación
            if None in membership:
                raise ValueError("Al menos un nodo no está asignado a ningún clúster.")

            # Calcular la modularidad usando igraph
            modularity_value = graph.modularity(membership)
            logging.info(f"Modularidad calculada exitosamente: {modularity_value}")
            return modularity_value

        except Exception as e:
            logging.error(f"Error al calcular la modularidad: {e}")
            return None

    # TODO: Gonzalo, porfi, cambia functional_enrichment_score para que reciva el grafo y los clústeres formados,
    #       para tener la funcionalidad del todo implementada aquí y no tener que hacer un enrichment en optimize.py

    @staticmethod
    def functional_enrichment_score(enrichment_data):
        """
        Calcula el puntaje promedio de enriquecimiento funcional para los clusters.

        :param enrichment_data: Diccionario donde las claves son los IDs de los clusters
                                y los valores son listas de p-valores de enriquecimiento para cada término.
        :return: Puntaje promedio de enriquecimiento funcional (log-transformado).
        """
        scores = []
        
        for cluster_id, p_values in enrichment_data.items():
            # Transformamos p-valores a un puntaje logarítmico
            log_scores = [-np.log10(p) if p > 0 else 0 for p in p_values]
            # Promedio del puntaje del cluster
            if log_scores:
                scores.append(sum(log_scores) / len(log_scores))
        
        return sum(scores) / len(scores) if scores else 0
