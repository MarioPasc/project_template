#!/usr/bin/env python3

import os
import pandas as pd
import gseapy as gp
from typing import Any, List, Dict, Union
import igraph


class FunctionalAnalysis:
    def __init__(self, graph: igraph.Graph):
        """
        Inicializa la clase con un grafo que contiene los clústeres.

        :param graph: Objeto grafo que ya incluye los clústeres y genes como nodos.
        """
        if not isinstance(graph, igraph.Graph):
            raise TypeError("El argumento 'graph' debe ser un objeto de tipo igraph.Graph.")
        self.graph = graph

    @staticmethod
    def _perform_enrichment_analysis(genes: List[str]) -> pd.DataFrame:
        """
        Realiza un análisis de enriquecimiento funcional para una lista de genes.

        :param genes: Lista de genes para los cuales realizar el enriquecimiento.
        :return: DataFrame con los resultados del análisis funcional.
        """
        try:
            if not genes:
                raise ValueError("La lista de genes está vacía.")

            # Realizar análisis funcional usando Enrichr
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets='GO_Biological_Process_2021',  # Base de datos funcional, ajustable
                organism='Human',  # Especificar organismo
                outdir=None,  # No guardar resultados en disco
                no_plot=True  # No generar gráficos
            )

            # Retornar resultados si hay datos
            if enr.results.empty:
                return pd.DataFrame()  # Retornar un DataFrame vacío si no hay resultados

            return enr.results

        except Exception as e:
            print(f"Error en _perform_enrichment_analysis para genes {genes}: {e}")
            return pd.DataFrame()

    def perform_analysis(self, clustering_data: Dict[str, Any], output_file: str, algorithm: str = None):
        """
        Realiza el análisis funcional para los clústeres extraídos de un JSON y guarda los resultados en un archivo CSV.

        :param clustering_data: Diccionario con la estructura del clustering.
        :param output_file: Ruta al archivo CSV donde se guardarán los resultados.
        :param algorithm: Nombre del algoritmo para el cual realizar el análisis. (Opcional)
        """
        try:
            if not isinstance(clustering_data, dict):
                raise ValueError("El argumento 'clustering_data' debe ser un diccionario.")

            all_results = []

            # Filtrar por algoritmo si se especifica
            if algorithm:
                if algorithm not in clustering_data:
                    raise ValueError(f"El algoritmo '{algorithm}' no está presente en los datos de clustering.")
                clustering_data = {algorithm: clustering_data[algorithm]}

            for key, clusters in clustering_data.items():
                for cluster_id, cluster_info in clusters.items():
                    genes = cluster_info.get("Genes", [])

                    print(f"Realizando análisis funcional para {key} - {cluster_id}...")
                    enrichment_results = self._perform_enrichment_analysis(genes)

                    if enrichment_results.empty:
                        print(f"No se encontraron resultados de enriquecimiento para {key} - {cluster_id}.")
                        continue

                    enrichment_results['Algorithm'] = key
                    enrichment_results['Cluster'] = cluster_id
                    all_results.append(enrichment_results)

            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                final_results.to_csv(output_file, index=False)
                print(f"Resultados guardados en {output_file}.")
            else:
                print("No se encontraron resultados significativos para los clústeres.")

        except Exception as e:
            print(f"Error en perform_analysis: {e}")

            
    def filter_results(self, input_file: str, output_file: str, 
                       p_value_threshold: float = None, 
                       combined_score_min: float = None):
        """
        Filtra los resultados del análisis funcional según p-valor o Combined Score.

        :param input_file: Ruta al archivo CSV con los resultados del análisis funcional.
        :param output_file: Ruta al archivo CSV donde se guardarán los resultados filtrados.
        :param p_value_threshold: Umbral máximo del p-valor. (Opcional)
        :param combined_score_min: Umbral mínimo de Combined Score. (Opcional)
        """
        try:
            # Cargar resultados
            data = pd.read_csv(input_file)

            # Validar que se proporcione al menos un criterio de filtrado
            if p_value_threshold is None and combined_score_min is None:
                raise ValueError("Debes proporcionar al menos un criterio de filtrado (p_value_threshold o combined_score_min).")

            # Aplicar filtros
            if p_value_threshold is not None:
                data = data[data['P-value'] <= p_value_threshold]

            if combined_score_min is not None:
                data = data[data['Combined Score'] >= combined_score_min]

            # Guardar resultados filtrados
            data.to_csv(output_file, index=False)
            print(f"Resultados filtrados guardados en {output_file}.")
        except Exception as e:
            print(f"Error al filtrar resultados: {e}")
