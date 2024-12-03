
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
        
def test_functional_analysis_with_filtering():
    """
    Prueba el análisis funcional con datos JSON y aplica filtrado por p-valor.
    """
    # Cargar el grafo desde un archivo
    network_file = "network.tsv"
    try:
        graph = network_to_igraph_format(network_file)
    except Exception as e:
        print(f"Error al cargar la red: {e}")
        return

    # JSON simplificado con clústeres
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

    # Instancia de la clase FunctionalAnalysis
    fa = FunctionalAnalysis(graph)

    # Paso 1: Ejecutar el análisis funcional
    analysis_output_file = "test_results.csv"
    try:
        fa.perform_analysis(clustering_data, analysis_output_file)
    except Exception as e:
        print(f"Error durante el análisis funcional: {e}")
        return

    # Paso 2: Filtrar los resultados por p-valor
    filtered_output_file = "filtered_results.csv"
    try:
        fa.filter_results(
            input_file=analysis_output_file,
            output_file=filtered_output_file,
            p_value_threshold=0.01  # Filtrar por p-valor menor o igual a 0.01
        )
        print(f"Filtrado completado. Resultados guardados en {filtered_output_file}.")
    except Exception as e:
        print(f"Error durante el filtrado: {e}")

# Ejecutar los tests
if __name__ == "__main__":
    test_functional_analysis_on_clusters()
    test_functional_analysis_with_filtering()
