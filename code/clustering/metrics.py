#!/usr/bin/env python3

import logging
from math import log10
from typing import List, Optional

import pandas as pd
import requests
from igraph import Graph

# import numpy as np
from stringdb import get_enrichment, get_string_ids


class Metrics:
    @staticmethod
    def modularity(
        graph: Graph, clusters: List[List[int]], logger: logging.Logger
    ) -> Optional[float]:
        """
        Calcula la modularidad (Q) usando igraph y registra eventos importantes en un archivo de registro.

        :param graph: Red como un objeto igraph.Graph.
        :param clusters: Lista de listas donde cada sublista representa un cluster con nodos.
        :param log_file: Ruta a un archivo de registro donde se escriben eventos importantes.
        :return: Un valor flotante representando la modularidad o None si ocurre un error.
        """
        try:
            # Validar que el grafo sea un objeto de tipo igraph.Graph
            if not isinstance(graph, Graph):
                raise TypeError(
                    "El parámetro 'graph' debe ser un objeto de tipo igraph.Graph."
                )

            # Validar que 'clusters' sea una lista de listas
            if not isinstance(clusters, list) or not all(
                isinstance(c, list) for c in clusters
            ):
                raise TypeError("El parámetro 'clusters' debe ser una lista de listas.")

            # Validar que todos los nodos en clusters estén en el rango de nodos del grafo
            num_nodes = graph.vcount()
            if any(
                node < 0 or node >= num_nodes
                for cluster in clusters
                for node in cluster
            ):
                raise ValueError(
                    f"Los nodos en 'clusters' deben estar en el rango 0 a {num_nodes - 1}."
                )

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
            logger.info(f"Modularidad calculada exitosamente: {modularity_value}")
            return modularity_value

        except Exception as e:
            logger.error(f"Error al calcular la modularidad: {e}")
            return None

    @staticmethod
    def functional_enrichment_score(
        graph: Graph, clusters: List[List[int]], logger: logging.Logger
    ) -> float:
        """
        Calcula una métrica de enriquecimiento funcional para los clústeres en un grafo.

        :param graph: Grafo como un objeto de igraph.
        :param clusters: Lista de listas, donde cada sublista representa un clúster con nodos.
        :param log_file: Ruta del archivo de logging donde se escribirán los resultados.
        :return: Valor flotante promedio del puntaje de enriquecimiento funcional para todos los clústeres.
        """
        # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            # Validar entradas
            Metrics._validate_inputs(graph, clusters)

            logger.info("Iniciando cálculo del puntaje de enriquecimiento funcional.")

            scores = []
            for i, cluster in enumerate(clusters):
                # Obtener genes del clúster
                genes = [
                    graph.vs[node]["name"]
                    for node in cluster
                    if "name" in graph.vs[node].attributes()
                ]
                if not genes:
                    logger.warning(f"Clúster {i} vacío o sin genes válidos. Se omite.")
                    continue

                # Enriquecimiento funcional para el clúster
                enrichment_results = Metrics._perform_enrichment(
                    genes=genes, logger=logger
                )
                if enrichment_results is None:
                    continue

                # Calcular el puntaje del clúster
                cluster_score = Metrics._calculate_cluster_score(
                    enriched_terms=enrichment_results, logger=logger
                )
                scores.append(cluster_score)
                logger.info(f"Puntaje para clúster {i}: {cluster_score}")

            # Calcular y retornar el promedio de los puntajes
            if scores:
                average_score = sum(scores) / len(scores)
                logger.info(
                    f"Puntaje promedio de enriquecimiento funcional: {average_score}"
                )
                return average_score
            else:
                logger.warning("No se calcularon puntajes válidos para los clústeres.")
                return 0.0

        except Exception as e:
            logger.error(f"Error en functional_enrichment_score: {e}")
            raise

    @staticmethod
    def _validate_inputs(graph: Graph, clusters: List[List[int]]) -> None:
        """Valida las entradas principales."""
        if not isinstance(graph, Graph):
            raise TypeError(
                "El parámetro 'graph' debe ser un objeto de tipo igraph.Graph."
            )
        if not isinstance(clusters, list) or not all(
            isinstance(c, list) for c in clusters
        ):
            raise TypeError("El parámetro 'clusters' debe ser una lista de listas.")

    @staticmethod
    def _perform_enrichment(genes: List[str], logger: logging.Logger) -> pd.DataFrame:
        """
        Realiza un análisis de enriquecimiento funcional para un conjunto de genes utilizando STRINGdb.

        :param genes: Lista de nombres de genes.
        :param log_file: Ruta del archivo de logging donde se registrarán los resultados.
        :return: DataFrame con los resultados del enriquecimiento funcional o None si no hay resultados.
        """

        try:
            # Validar entrada
            if not isinstance(genes, list) or not all(
                isinstance(gene, str) for gene in genes
            ):
                raise ValueError(
                    "El argumento 'genes' debe ser una lista de nombres de genes en formato string."
                )

            logger.info(
                f"Realizando análisis de enriquecimiento para los genes: {genes}"
            )

            # Obtener IDs de STRINGdb
            string_ids = get_string_ids(genes)
            if string_ids.empty:
                logger.warning(
                    "No se encontraron IDs válidos para los genes proporcionados."
                )
                return None

            logger.info(f"IDs de STRINGdb obtenidos: {string_ids['stringId'].tolist()}")

            # Realizar análisis de enriquecimiento
            enrichment_results = get_enrichment(string_ids["stringId"].tolist())
            filtered_terms = enrichment_results[
                enrichment_results["category"] == "Process"
            ]
            if filtered_terms.empty:
                logger.warning(
                    "No se encontraron términos enriquecidos para los genes proporcionados."
                )
                return None

            logger.info(
                f"Resultados de enriquecimiento obtenidos: {len(filtered_terms)} términos enriquecidos."
            )
            return filtered_terms

        except Exception as e:
            logger.error(f"Error durante el análisis de enriquecimiento funcional: {e}")
            raise

    @staticmethod
    def _get_go_term_depth(go_id: str, logger: logging.Logger) -> int:
        """
        Consulta la profundidad de un término GO en la jerarquía Gene Ontology usando la API de GO.

        :param go_id: Identificador del término GO (por ejemplo, "GO:0008150").
        :return: Entero que representa la profundidad del término en la ontología.
        """
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}/ancestors"
        try:
            response = requests.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            data = response.json()

            # Contar la profundidad como el número de ancestros hasta la raíz
            ancestors = data.get("results", [])[0].get("ancestors", [])
            return len(ancestors)
        except Exception as e:
            logger.warning(f"No se pudo obtener la profundidad para {go_id}: {e}")
            return 1  # Profundidad predeterminada si ocurre un error

    @staticmethod
    def _calculate_cluster_score(
        enriched_terms: pd.DataFrame, logger: logging.Logger
    ) -> float:
        """
        Calcula el puntaje combinado para un clúster a partir de los términos enriquecidos.

        :param enriched_terms: DataFrame con resultados del enriquecimiento funcional.
        :return: Puntaje combinado para el clúster.
        """
        try:
            scores = []
            for _, row in enriched_terms.iterrows():
                p_value: Optional[float] = row.get("p_value", None)
                go_id: Optional[str] = row.get(
                    "term", None
                )  # Asumiendo que la columna 'term' contiene el ID GO

                if (
                    p_value is None
                    or not isinstance(p_value, (int, float))
                    or p_value <= 0
                ):
                    continue

                # Obtener la profundidad del término GO
                depth = (
                    Metrics._get_go_term_depth(go_id=go_id, logger=logger)
                    if go_id
                    else 1
                )

                # Calcular score: -log10(p_value) * depth
                scores.append(
                    -log10(p_value) * depth / 600
                )  # Max -log10(p-value) = 60 Max depth = 10

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Error en _calculate_cluster_score: {e}")
            raise
