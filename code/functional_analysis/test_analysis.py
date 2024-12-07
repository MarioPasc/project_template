import json
from functional_analysis import FunctionalAnalysis

# from clustering.utils import network_to_igraph_format
from utils import network_to_igraph_format


def test_functional_analysis_on_algorithm():
    """
    Prueba el análisis funcional especificando un algoritmo.
    """
    # Cargar el grafo desde un archivo
    network_file = "network.tsv"
    try:
        graph = network_to_igraph_format(network_file)
    except Exception as e:
        print(f"Error al cargar la red: {e}")
        return

    # JSON de prueba con múltiples algoritmos
    clustering_data = {
        "leiden_max_modularity": {
            "Cluster 1": {"Genes": ["GRN", "CHMP2B", "MAPT"], "Nodes": [0, 3, 4]},
            "Cluster 2": {"Genes": ["FUS", "TARDBP", "TAF15"], "Nodes": [1, 7, 25]},
        },
        "multilevel_max_modularity": {
            "Cluster 1": {"Genes": ["APP", "PSEN1", "PSEN2"], "Nodes": [5, 6, 7]},
            "Cluster 2": {"Genes": ["MAPT", "TUBA4A", "VCP"], "Nodes": [8, 9, 10]},
        },
    }

    # Instancia de la clase FunctionalAnalysis
    fa = FunctionalAnalysis(graph)

    # Paso 1: Ejecutar el análisis funcional para un algoritmo específico
    output_file_specific = "results_leiden.csv"
    try:
        fa.perform_analysis(
            clustering_data=clustering_data,
            output_file=output_file_specific,
            algorithm="leiden_max_modularity",  # Especificar algoritmo
        )
        print(f"Análisis completado. Resultados guardados en {output_file_specific}.")
    except Exception as e:
        print(f"Error durante el análisis funcional para un algoritmo específico: {e}")


def test_functional_analysis_with_filtering_on_algorithm():
    """
    Prueba el análisis funcional para un algoritmo específico y aplica filtrado por p-valor.
    """
    # Cargar el grafo desde un archivo
    network_file = "network.tsv"
    try:
        graph = network_to_igraph_format(network_file)
    except Exception as e:
        print(f"Error al cargar la red: {e}")
        return

    # JSON de prueba con múltiples algoritmos
    clustering_data = {
        "leiden_max_modularity": {
            "Cluster 1": {"Genes": ["GRN", "CHMP2B", "MAPT"], "Nodes": [0, 3, 4]},
            "Cluster 2": {"Genes": ["FUS", "TARDBP", "TAF15"], "Nodes": [1, 7, 25]},
        },
        "multilevel_max_modularity": {
            "Cluster 1": {"Genes": ["APP", "PSEN1", "PSEN2"], "Nodes": [5, 6, 7]},
            "Cluster 2": {"Genes": ["MAPT", "TUBA4A", "VCP"], "Nodes": [8, 9, 10]},
        },
    }

    # Instancia de la clase FunctionalAnalysis
    fa = FunctionalAnalysis(graph)

    # Paso 1: Ejecutar el análisis funcional para un algoritmo específico
    analysis_output_file = "results_leiden.csv"
    try:
        fa.perform_analysis(
            clustering_data=clustering_data,
            output_file=analysis_output_file,
            algorithm="leiden_max_modularity",  # Especificar algoritmo
        )
    except Exception as e:
        print(f"Error durante el análisis funcional: {e}")
        return

    # Paso 2: Filtrar los resultados por p-valor
    filtered_output_file = "filtered_results_leiden.csv"
    try:
        fa.filter_results(
            input_file=analysis_output_file,
            output_file=filtered_output_file,
            p_value_threshold=0.01,  # Filtrar por p-valor menor o igual a 0.01
        )
        print(f"Filtrado completado. Resultados guardados en {filtered_output_file}.")
    except Exception as e:
        print(f"Error durante el filtrado: {e}")


# Ejecutar los tests
if __name__ == "__main__":
    test_functional_analysis_on_algorithm()
    test_functional_analysis_with_filtering_on_algorithm()
