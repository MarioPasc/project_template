
import json
from functional_analysis import FunctionalAnalysis
# from clustering.utils import network_to_igraph_format
from utils import network_to_igraph_format

def test_functional_analysis_on_clusters():
    """
    Prueba el análisis funcional sobre los clústeres extraídos del grafo (igraph).
    """
    # Ruta al archivo de red
    network_file = "network.tsv"

    # Convertir la red al formato igraph
    try:
        graph = network_to_igraph_format(network_file)
    except Exception as e:
        print(f"Error al cargar la red: {e}")
        return

    # Definir clústeres de prueba
    clustering_data = {
        "leiden_max_modularity": {
            "Cluster 1": {
                "Genes": ["GRN", "CHMP2B", "MAPT"],
                "Nodes": [0, 3, 4]
            },
            "Cluster 2": {
                "Genes": ["FUS", "TARDBP", "TAF15"],
                "Nodes": [1, 7, 25]
            }
        }
    }

    # Crear instancia de FunctionalAnalysis con el grafo igraph
    fa = FunctionalAnalysis(graph)

    # Definir la ruta de salida
    output_file = "test_results.csv"
    
    try:
        # Ejecutar el análisis funcional
        fa.perform_analysis(clustering_data, output_file)
        print(f"Prueba completada. Resultados guardados en {output_file}.")
    except Exception as e:
        print(f"Error durante el análisis funcional: {e}")

# Ejecutar el test
if __name__ == "__main__":
    test_functional_analysis_on_clusters()
