#!/usr/bin/env python3

from math import log10
from igraph import Graph
import logging
#import numpy as np
from stringdb import get_string_ids, get_enrichment
from typing import List, Optional
import pandas as pd

class Metrics:
    @staticmethod
    def modularity(graph: Graph, clusters: List[List[int]], log_file: Optional[str] = None) -> Optional[float]:
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

    @staticmethod
    def functional_enrichment_score(graph: Graph, clusters: List[List[int]], log_file: Optional[str]) -> float:
        """
        Calcula una métrica de enriquecimiento funcional para los clústeres en un grafo.

        :param graph: Grafo como un objeto de igraph.
        :param clusters: Lista de listas, donde cada sublista representa un clúster con nodos.
        :param log_file: Ruta del archivo de logging donde se escribirán los resultados.
        :return: Valor flotante promedio del puntaje de enriquecimiento funcional para todos los clústeres.
        """
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Validar entradas
            Metrics._validate_inputs(graph, clusters)

            logging.info("Iniciando cálculo del puntaje de enriquecimiento funcional.")

            scores = []
            for i, cluster in enumerate(clusters):
                # Obtener genes del clúster
                genes = [graph.vs[node]["name"] for node in cluster if "name" in graph.vs[node].attributes()]
                if not genes:
                    logging.warning(f"Clúster {i} vacío o sin genes válidos. Se omite.")
                    continue

                # Enriquecimiento funcional para el clúster
                enrichment_results = Metrics._perform_enrichment(genes, log_file)
                if enrichment_results is None:
                    continue

                # Calcular el puntaje del clúster
                cluster_score = Metrics._calculate_cluster_score(enrichment_results)
                scores.append(cluster_score)
                logging.info(f"Puntaje para clúster {i}: {cluster_score}")

            # Calcular y retornar el promedio de los puntajes
            if scores:
                average_score = sum(scores) / len(scores)
                logging.info(f"Puntaje promedio de enriquecimiento funcional: {average_score}")
                return average_score
            else:
                logging.warning("No se calcularon puntajes válidos para los clústeres.")
                return 0.0

        except Exception as e:
            logging.error(f"Error en functional_enrichment_score: {e}")
            raise

    @staticmethod
    def _validate_inputs(graph: Graph, clusters: List[List[int]]) -> None:
        """Valida las entradas principales."""
        if not isinstance(graph, Graph):
            raise TypeError("El parámetro 'graph' debe ser un objeto de tipo igraph.Graph.")
        if not isinstance(clusters, list) or not all(isinstance(c, list) for c in clusters):
            raise TypeError("El parámetro 'clusters' debe ser una lista de listas.")

    @staticmethod
    def _perform_enrichment(genes: List[str], log_file: str) -> pd.DataFrame:
        """
        Realiza un análisis de enriquecimiento funcional para un conjunto de genes utilizando STRINGdb.

        :param genes: Lista de nombres de genes.
        :param log_file: Ruta del archivo de logging donde se registrarán los resultados.
        :return: DataFrame con los resultados del enriquecimiento funcional o None si no hay resultados.
        """
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Validar entrada
            if not isinstance(genes, list) or not all(isinstance(gene, str) for gene in genes):
                raise ValueError("El argumento 'genes' debe ser una lista de nombres de genes en formato string.")

            logging.info(f"Realizando análisis de enriquecimiento para los genes: {genes}")

            # Obtener IDs de STRINGdb
            string_ids = get_string_ids(genes)
            if string_ids.empty:
                logging.warning("No se encontraron IDs válidos para los genes proporcionados.")
                return None
            
            logging.info(f"IDs de STRINGdb obtenidos: {string_ids['stringId'].tolist()}")

            # Realizar análisis de enriquecimiento
            enrichment_results = get_enrichment(string_ids['stringId'].tolist())
            if enrichment_results.empty:
                logging.warning("No se encontraron términos enriquecidos para los genes proporcionados.")
                return None

            logging.info(f"Resultados de enriquecimiento obtenidos: {len(enrichment_results)} términos enriquecidos.")
            return enrichment_results

        except Exception as e:
            logging.error(f"Error durante el análisis de enriquecimiento funcional: {e}")
            raise


    @staticmethod
    def _calculate_cluster_score(enriched_terms: pd.DataFrame) -> float:
        """
        Calcula el puntaje combinado para un clúster a partir de los términos enriquecidos.

        :param enriched_terms: DataFrame con resultados del enriquecimiento funcional.
        :return: Puntaje combinado para el clúster.
        """
        try:
            scores = []
            for _, row in enriched_terms.iterrows():
                p_value: Optional[float] = row.get('p_value', None)
                depth: int = row.get('depth', 1)  # Por defecto, profundidad 1

                if p_value is None or not isinstance(p_value, (int, float)) or p_value <= 0:
                    continue

                # Calcular score: -log10(p_value) * depth
                scores.append(-log10(p_value) * depth)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logging.error(f"Error en _calculate_cluster_score: {e}")
            raise
