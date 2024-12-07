import pandas as pd
from visualization import FunctionalVisualization, prepare_data_for_visualization_from_df

def test_visualizations():
    """
    Prueba la generación de todas las gráficas y guarda cada una en un archivo PDF.
    """

    # Rutas de los archivos CSV
    file_enrichment = "results_leiden_max_enrichment.csv"
    file_modularity = "results_leiden_max_modularity.csv"

    # Leer datos de enrichment para las pruebas de otras gráficas
    df_enrichment = pd.read_csv(file_enrichment)

    # Preparar datos para las visualizaciones
    prepared_df, gene_sets = prepare_data_for_visualization_from_df(df_enrichment)

    # Verificar que los datos se prepararon correctamente
    if prepared_df is None or gene_sets is None:
        raise ValueError("Error al preparar los datos para las visualizaciones.")

    # Rutas para guardar los gráficos
    output_dot_plot = "dot_plot.pdf"
    output_bar_plot = "bar_plot.pdf"
    output_cnetplot = "cnetplot.pdf"
    output_upset_plot = "upset_plot.pdf"
    output_venn_diagram = "venn_diagram.pdf"

    # Generar y guardar las gráficas
    try:
        print("Generando Dot Plot...")
        FunctionalVisualization.dot_plot(prepared_df, output_file=output_dot_plot)

        print("Generando Bar Plot...")
        FunctionalVisualization.bar_plot(prepared_df, output_file=output_bar_plot)

        print("Generando Cnetplot...")
        FunctionalVisualization.cnet_plot(prepared_df, gene_sets, output_file=output_cnetplot)

        print("Generando UpSet Plot...")
        FunctionalVisualization.upset_plot(df_enrichment, output_file=output_upset_plot)

        print("Generando Venn Diagram...")
        FunctionalVisualization.venn_diagram(
            file_modularity=file_modularity,
            file_enrichment=file_enrichment,
            output_file=output_venn_diagram
        )

        print("Todas las gráficas se han guardado correctamente.")
    except Exception as e:
        print(f"Error durante la generación de gráficas: {e}")

# Ejecutar la prueba
if __name__ == "__main__":
    test_visualizations()