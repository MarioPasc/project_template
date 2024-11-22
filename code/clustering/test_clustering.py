from igraph import Graph
from algorithms import Algorithms
from metrics import Metrics
import random
# Crear un grafo de ejemplo
#g = Graph.Erdos_Renyi(n=50, m=100)

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

g, _ = create_test_graph()

# Realizar el clustering con el algoritmo multilevel
clusters = Algorithms.multilevel_clustering(g)

# Convertir el objeto VertexClustering en una lista de listas
cluster_list = Algorithms.convert_clustering_to_list(clusters)

# Ahora puedes usar cluster_list con las métricas
log_file = "metrics_log.txt"

# Calcular el Functional Enrichment Score
score = Metrics.functional_enrichment_score(g, cluster_list, log_file)
print(f"Puntaje de enriquecimiento funcional: {score}")

# Calcular la modularidad
modularity_score = Metrics.modularity(g, cluster_list, log_file=log_file)
print(f"Modularidad: {modularity_score}")
