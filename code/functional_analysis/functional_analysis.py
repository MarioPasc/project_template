
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
                gene_sets='KEGG_2021_Human',  # Base de datos funcional, ajustable
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

    def perform_analysis(self, clustering_data: Dict[str, Any], output_file: str):
        """
        Realiza el análisis funcional para los clústeres extraídos de un JSON.
        y guarda los resultados en un archivo CSV.

        :param clustering_data: Diccionario con la estructura del clustering.
        :param output_file: Ruta al archivo CSV donde se guardarán los resultados.
        """
        try:
            if not isinstance(clustering_data, dict):
                raise ValueError("El argumento 'clustering_data' debe ser un diccionario.")

            all_results = []

            for key, clusters in clustering_data.items():
                for cluster_id, cluster_info in clusters.items():
                    genes = cluster_info.get("Genes", [])
                    if len(genes) < 3:  # Ignorar clústeres con menos de 3 genes
                        print(f"{key} - {cluster_id} ignorado (menos de 3 genes).")
                        continue

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
            print(f"Error en perform_analysis_from_json: {e}")

