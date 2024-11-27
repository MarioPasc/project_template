
import networkx as nx
import pandas as pd
import gseapy as gp

class FunctionalAnalysis:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @staticmethod
    def _perform_enrichment_analysis(genes: list) -> pd.DataFrame:
        """
        Realiza un análisis de enriquecimiento funcional para una lista de genes.
        """
        # Realizar el análisis de enriquecimiento utilizando GSEApy
        enr = gp.enrichr(gene_list=genes,
                         gene_sets='KEGG_2021_Human',  # Puedes elegir otras bases de datos
                         organism='Human',  # Especifica el organismo
                         outdir=None,  # No guardar resultados en disco
                         no_plot=True)  # No generar gráficos

        # Verificar si se obtuvieron resultados
        if enr.results.empty:
            return pd.DataFrame()  # Retornar un DataFrame vacío si no hay resultados

        # Retornar los resultados como un DataFrame
        return enr.results

    def perform_analysis(self, output_file: str):
        """
        Extrae los clústeres del grafo, realiza el análisis de enriquecimiento
        para cada clúster y guarda los resultados en un archivo CSV.
        """
        # Extraer los clústeres del grafo
        clusters = self._extract_clusters()

        # Lista para almacenar todos los resultados
        all_results = []

        # Iterar sobre cada clúster y realizar el análisis de enriquecimiento
        for cluster_id, genes in clusters.items():
            if len(genes) < 3:
                # Omitir clústeres con menos de 3 genes
                continue

            # Realizar el análisis de enriquecimiento
            enrichment_results = self._perform_enrichment_analysis(genes)

            # Agregar una columna para identificar el clúster
            enrichment_results['Cluster'] = cluster_id

            # Agregar los resultados a la lista
            all_results.append(enrichment_results)

        # Concatenar todos los resultados en un solo DataFrame
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            # Guardar los resultados en un archivo CSV
            final_results.to_csv(output_file, index=False)
        else:
            print("No se encontraron resultados de enriquecimiento para los clústeres proporcionados.")

    def _extract_clusters(self) -> dict:
        """
        Extrae los clústeres del grafo y los retorna como un diccionario.
        """
        # Suponiendo que los nodos del grafo tienen un atributo 'cluster' que indica su clúster
        clusters = {}
        for node, data in self.graph.nodes(data=True):
            cluster_id = data.get('cluster')
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node)
        return clusters
