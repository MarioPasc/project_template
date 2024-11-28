
import os
import pandas as pd
import gseapy as gp
from typing import List, Dict, Union
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

    def perform_analysis(self, clusters: List[List[int]], output_file: str):
        """
        Realiza el análisis funcional para los clústeres extraídos del grafo
        y guarda los resultados en un archivo CSV.

        :param clusters: Lista de listas, donde cada sublista contiene índices de nodos que pertenecen a un clúster.
        :param output_file: Ruta al archivo CSV donde se guardarán los resultados.
        """
        try:
            # Validar la entrada
            if not isinstance(clusters, list) or not all(isinstance(c, list) for c in clusters):
                raise ValueError("El parámetro 'clusters' debe ser una lista de listas.")
            if not isinstance(output_file, str) or not output_file.endswith('.csv'):
                raise ValueError("El parámetro 'output_file' debe ser una cadena que termine en '.csv'.")

            # Extraer los clústeres del grafo
            cluster_dict = self._extract_clusters(clusters)

            if not cluster_dict:
                raise ValueError("No se encontraron clústeres válidos en el grafo.")

            # Lista para almacenar resultados de todos los clústeres
            all_results = []

            # Iterar sobre cada clúster y realizar análisis funcional
            for cluster_id, genes in cluster_dict.items():
                try:
                    if len(genes) < 3:  # Ignorar clústeres con menos de 3 genes
                        print(f"Clúster {cluster_id} ignorado (menos de 3 genes).")
                        continue

                    print(f"Realizando análisis funcional para el clúster {cluster_id}...")
                    enrichment_results = self._perform_enrichment_analysis(genes)

                    if enrichment_results.empty:
                        print(f"No se encontraron resultados de enriquecimiento para el clúster {cluster_id}.")
                        continue

                    # Agregar una columna para identificar el clúster
                    enrichment_results['Cluster'] = cluster_id
                    all_results.append(enrichment_results)

                except Exception as e:
                    print(f"Error al analizar el clúster {cluster_id}: {e}")
                    continue

            # Concatenar todos los resultados y guardarlos en un CSV
            if all_results:
                final_results = pd.concat(all_results, ignore_index=True)
                final_results.to_csv(output_file, index=False)
                print(f"Resultados guardados en {output_file}.")
            else:
                print("No se encontraron resultados significativos para los clústeres.")

        except Exception as e:
            print(f"Error en perform_analysis: {e}")

    def _extract_clusters(self, clusters: List[List[int]]) -> Dict[str, List[str]]:
        """
        Extrae los clústeres como un diccionario de genes basado en la lista de índices y el grafo.

        :param clusters: Lista de listas, donde cada sublista contiene índices de nodos que pertenecen a un clúster.
        :return: Diccionario donde las claves son los IDs de los clústeres (Cluster1, Cluster2, ...)
                y los valores son listas de genes correspondientes a cada clúster.
        """
        try:
            cluster_dict = {}
            for i, cluster in enumerate(clusters):
                cluster_id = f"Cluster{i + 1}"
                genes = [
                    self.graph.vs[node]["name"]  # Convertir índices en nombres de genes
                    for node in cluster
                    if "name" in self.graph.vs[node].attributes()
                ]
                if genes:  # Solo agregar si hay genes válidos
                    cluster_dict[cluster_id] = genes
                else:
                    print(f"Advertencia: El clúster {cluster_id} no tiene genes válidos.")

            if not cluster_dict:
                print("Advertencia: No se encontraron genes válidos en los clústeres.")

            return cluster_dict

        except Exception as e:
            print(f"Error en _extract_clusters: {e}")
            return {}
