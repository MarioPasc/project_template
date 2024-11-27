import networkx as nx
import os
import pandas as pd
from functional_analysis import FunctionalAnalysis 

# Crear un grafo simulado para pruebas
def create_test_graph():
    """
    Crea un grafo de prueba con nodos y atributos de clúster.
    """
    G = nx.Graph()
    
    # Añadir nodos con atributos de clúster
    G.add_node('TP53', cluster='Cluster1')
    G.add_node('BRCA1', cluster='Cluster1')
    G.add_node('MDM2', cluster='Cluster1')
    G.add_node('ERK2', cluster='Cluster2')
    G.add_node('MEK1', cluster='Cluster2')
    G.add_node('AKT1', cluster='Cluster3')  # Clúster pequeño para probar exclusión
    
    # Añadir conexiones (no son relevantes para los clústeres, pero válidas para la estructura del grafo)
    G.add_edges_from([('TP53', 'BRCA1'), ('TP53', 'MDM2'), ('ERK2', 'MEK1')])
    
    return G

# Probar la clase FunctionalAnalysis
def test_functional_analysis():
    """
    Prueba el flujo completo de la clase FunctionalAnalysis.
    """
    # Crear el grafo de prueba
    graph = create_test_graph()

    # Crear una instancia de FunctionalAnalysis
    fa = FunctionalAnalysis(graph)

    # Definir el archivo de salida para los resultados
    output_file = 'test_results.csv'

    # Ejecutar el análisis
    print("Ejecutando el análisis funcional...")
    fa.perform_analysis(output_file)

    # Verificar que se haya generado el archivo de resultados
    if os.path.exists(output_file):
        print(f"Archivo de resultados generado correctamente: {output_file}")
        
        # Leer el archivo y verificar su contenido
        results = pd.read_csv(output_file)
        print("Contenido del archivo de resultados:")
        print(results.head())
        
        # Limpieza: eliminar el archivo de prueba después de validar
        os.remove(output_file)
    else:
        print("Error: No se generó el archivo de resultados.")

# Ejecutar el test
if __name__ == "__main__":
    test_functional_analysis()