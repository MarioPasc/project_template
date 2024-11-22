'''
import sys
sys.path.append('../Py_libs')

from igraph import Graph
from metrics import Metrics

# Crear la red con igraph
edges = [(0, 1), (0, 2), (1, 2),  # Clúster 1
         (3, 4), (3, 5), (4, 5),  # Clúster 2
         (2, 3)]  # Conexión entre clústeres
graph = Graph(edges=edges)

# Clústeres representados como listas de nodos
clusters = [[0, 1, 2], [3, 4, 5]]

# Datos de enriquecimiento funcional simulados
enrichment_data = {
    0: [0.001, 0.05, 0.01],  # p-valores para cluster 0
    1: [0.02, 0.03]          # p-valores para cluster 1
}


log_file = "../logs/clustering.log"

# Cálculo de métricas
modularity_score = Metrics.modularity(graph, clusters, log_file=log_file)
#functional_enrichment_score = Metricas.puntuaje_enriquecimiento_funcional(enrichment_data)

print(f"Modularidad: {modularity_score}")
#print(f"Puntaje de Enriquecimiento Funcional: {functional_enrichment_score}")
'''

from igraph import Graph
from metrics import Metrics

# Crear un grafo de ejemplo con nombres de genes reales
genes = ["TP53", "BRCA1", "EGFR", "MYC", "CDK2"]
edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
graph = Graph(edges=edges)
graph.vs["name"] = genes

# Definir clusters (nodos indexados)
clusters = [[0, 1], [2, 3, 4]]

# Ruta para el log
log_file = "../logs/enrichment.log"

# Llamar a la función
score = Metrics.functional_enrichment_score(graph, clusters, log_file)
print(f"Puntaje de enriquecimiento funcional: {score}")

modularity_score = Metrics.modularity(graph, clusters, log_file=log_file)
print(f"Modularidad: {modularity_score}")
