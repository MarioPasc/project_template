
import networkx as nx
import pandas as pd
import gseapy as gp
from typing import List, Dict

class FunctionalAnalysis:
    def __init__(self, graph: nx.Graph):
        """
        Inicializa la clase con un grafo que contiene los clústeres.

        :param graph: Objeto grafo que ya incluye los clústeres y genes como nodos.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("El argumento 'graph' debe ser un objeto de tipo networkx.Graph.")
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
            # Captura cualquier error durante el análisis funcional
            print(f"Error en _perform_enrichment_analysis para genes {genes}: {e}")
            return pd.DataFrame()

    def perform_analysis(self, output_file: str):
        """
        Extrae los clústeres del grafo, realiza análisis funcional para cada clúster
        y guarda los resultados en un archivo CSV.

        :param output_file: Ruta al archivo CSV donde se guardarán los resultados.
        """
        try:
            # Validar la entrada
            if not isinstance(output_file, str) or not output_file.endswith('.csv'):
                raise ValueError("El parámetro 'output_file' debe ser una cadena que termine en '.csv'.")

            # Extraer los clústeres del grafo
            clusters = self._extract_clusters()

            if not clusters:
                raise ValueError("No se encontraron clústeres en el grafo.")

            # Lista para almacenar resultados de todos los clústeres
            all_results = []

            # Iterar sobre cada clúster y realizar análisis funcional
            for cluster_id, genes in clusters.items():
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

    def _extract_clusters(self) -> dict:
        """
        Extrae los clústeres del grafo como un diccionario.

        :return: Diccionario donde las claves son los IDs de los clústeres y los valores son listas de genes.
        """
        try:
            clusters = {}
            for node, data in self.graph.nodes(data=True):
                cluster_id = data.get('cluster')  # Suponiendo que los nodos tienen el atributo 'cluster'
                if cluster_id:
                    clusters.setdefault(cluster_id, []).append(node)

            if not clusters:
                print("Advertencia: No se encontraron nodos con clústeres en el grafo.")
            return clusters

        except Exception as e:
            print(f"Error en _extract_clusters: {e}")
            return {}
