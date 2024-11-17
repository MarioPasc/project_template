#!/usr/bin/env python3

import igraph as ig
import networkx as nx
import numpy as np

class Metricas:
    @staticmethod
    def modularidad(graph, clusters):
        """
        Calcula la modularidad (Q) usando igraph.

        :param graph: Red como un objeto igraph.Graph.
        :param clusters: Lista de listas donde cada sublista representa un cluster con nodos.
        :return: Un valor flotante representando la modularidad.
        """
        # Convertir los clústeres a una lista de asignaciones (membership)
        membership = [None] * graph.vcount()
        for cluster_index, cluster in enumerate(clusters):
            for node in cluster:
                membership[node] = cluster_index

        # Calcular la modularidad usando igraph
        return graph.modularity(membership)

    @staticmethod
    def puntuaje_enriquecimiento_funcional(enrichment_data):
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
