from igraph import Graph
import random
import logging
from metrics import Metrics

# Crear grafo y clústeres
def create_test_graph():
    # Ejemplo de nombres de genes en distintas comunidades
    genes_cluster_1 = ["TP53", "BRCA1", "MDM2", "CDK2", "EGFR"]
    genes_cluster_2 = ["MYC", "FOXO3", "PIK3CA", "AKT1", "GSK3B"]
    genes_cluster_3 = ["MAPK1", "ERK2", "RAF1", "MEK1", "BRAF"]
    all_genes = genes_cluster_1 + genes_cluster_2 + genes_cluster_3

    # Crear grafo
    g = Graph()
    g.add_vertices(len(all_genes))
    g.vs["name"] = all_genes  # Asignar nombres a los nodos

    # Añadir interacciones dentro de clústeres
    for cluster in [genes_cluster_1, genes_cluster_2, genes_cluster_3]:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if random.random() < 0.8:  # Probabilidad alta intra-clúster
                    g.add_edge(all_genes.index(cluster[i]), all_genes.index(cluster[j]))

    # Añadir conexiones inter-clúster
    for i in range(len(genes_cluster_1)):
        if random.random() < 0.2:  # Probabilidad baja inter-clúster
            g.add_edge(all_genes.index(genes_cluster_1[i]), all_genes.index(genes_cluster_2[i % len(genes_cluster_2)]))

    for i in range(len(genes_cluster_2)):
        if random.random() < 0.2:
            g.add_edge(all_genes.index(genes_cluster_2[i]), all_genes.index(genes_cluster_3[i % len(genes_cluster_3)]))

    return g, [genes_cluster_1, genes_cluster_2, genes_cluster_3]

# Archivo de log
log_file = "../logs/test_metrics.log"
logging.basicConfig(level=logging.INFO, filename=log_file, format="%(asctime)s - %(levelname)s - %(message)s")

# Crear el grafo y los clústeres
graph, clusters = create_test_graph()

# Convertir clústeres a índices
clusters_as_indices = [
    [graph.vs.find(name=gene).index for gene in cluster] for cluster in clusters
]

# Calcular el puntaje de enriquecimiento funcional
try:
    score = Metrics.functional_enrichment_score(graph, clusters_as_indices, log_file)
    print(f"Puntaje de enriquecimiento funcional: {score}")
except Exception as e:
    print(f"Error al calcular el puntaje de enriquecimiento funcional: {e}")

# Calcular la modularidad
try:
    modularity_score = Metrics.modularity(graph, clusters_as_indices, log_file=log_file)
    print(f"Modularidad: {modularity_score}")
except Exception as e:
    print(f"Error al calcular la modularidad: {e}")


#print(Metrics._get_go_term_depth("GO:0051052"))
