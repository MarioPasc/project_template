from clustering.utils import network_to_igraph_format
from functional_analysis import FunctionalAnalysis 

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
    clusters = [
        [0, 5, 8, 9, 10, 11, 13, 17, 21, 22, 24, 25, 37, 38, 41, 47],
        [1, 3, 4, 6, 12, 15, 16, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 46, 48],
        [2, 7, 14, 18, 19, 20, 42, 43, 44, 45, 49]
    ]

    # Crear instancia de FunctionalAnalysis con el grafo igraph
    fa = FunctionalAnalysis(graph)

    # Definir la ruta de salida
    output_file = "test_results.csv"
    
    try:
        # Ejecutar el análisis funcional
        fa.perform_analysis(clusters, output_file)
        print(f"Prueba completada. Resultados guardados en {output_file}.")
    except Exception as e:
        print(f"Error durante el análisis funcional: {e}")

# Ejecutar el test
if __name__ == "__main__":
    test_functional_analysis_on_clusters()

