#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Optional
from upsetplot import UpSet
from matplotlib_venn import venn2
import igraph as ig
from matplotlib.lines import Line2D

import scienceplots

# Style
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "300"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


class FunctionalVisualization:
    """
    Clase para generar visualizaciones relacionadas con el análisis funcional.
    
    Incluye las siguientes funciones principales:
    - prepare_data_for_visualization_from_df:
        Prepara los datos para visualizaciones basadas en un DataFrame con resultados de análisis funcional.
        Convierte el formato de columnas para cálculos y genera diccionarios para graficar relaciones gene-término.

    - Dot Plot: Muestra la significancia y magnitud del enriquecimiento.
    
    - Bar Plot: Resalta los términos más significativos.
    
    - Cnetplot: Visualiza las relaciones entre genes y categorías enriquecidas.
    
    - upset_plot:
        Genera un gráfico UpSet que analiza las intersecciones y patrones entre términos enriquecidos y clusters.
        Este tipo de gráfico es útil para identificar patrones solapados entre conjuntos de datos funcionales.

    - venn_diagram:
        Genera un diagrama de Venn que compara dos conjuntos de términos enriquecidos provenientes
        de diferentes métodos de clustering o análisis.
    """

    @staticmethod
    def prepare_data_for_visualization_from_df(df: pd.DataFrame):
        """
        Prepara los datos del DataFrame generado por FunctionalAnalysis para las gráficas.

        :param df: DataFrame con los resultados del análisis funcional.

        :return:
            - DataFrame procesado con columnas adicionales ['Observed', 'Total', 'Gene Ratio'] para dot_plot, bar_plot y cnetplot.
            - Diccionario donde las claves son términos ('Term') y los valores son listas de genes ('Genes') para cnetplot.
        """
        try:
            # Convertir Overlap a Gene Ratio (proporción)
            df[["Observed", "Total"]] = (
                df["Overlap"].str.split("/", expand=True).astype(int)
            )
            df["Gene Ratio"] = df["Observed"] / df["Total"]

            # Crear un diccionario para cnetplot
            gene_sets = {
                row["Term"]: row["Genes"].split(", ") for _, row in df.iterrows()
            }

            return df, gene_sets

        except Exception as e:
            print(f"Error en prepare_data_for_visualization_from_df: {e}")
            return None, None

    @staticmethod
    def dot_plot(df: pd.DataFrame, output_file: str = None):
        """
        Genera un gráfico de puntos (Dot Plot) basado en términos enriquecidos.

        :param df: DataFrame con columnas ['Term', 'Adjusted P-value', 'Gene Ratio'].
        :param output_file: Ruta para guardar el gráfico (opcional).
        """
        try:
            # Ordenar los términos de significancia (los más significativos primero)
            df = df.sort_values("Adjusted P-value", ascending=True).head(20)

            # Crear un gráfico de puntos
            plt.figure(figsize=(10, 8))
            scatter = sns.scatterplot(
                data=df,
                x="Gene Ratio",
                y="Term",
                size="Adjusted P-value",
                hue="Adjusted P-value",
                sizes=(300, 50),  # Aumenta el rango de tamaños de los puntos
                palette="coolwarm",  # Mejorar contraste de colores
                legend="brief",
            )
            plt.title("Dot Plot - Enrichment Analysis")
            plt.xlabel("Gene Ratio")
            plt.ylabel("Term")
            #plt.gca().yaxis.set_tick_params(labelsize=10)  # Mejorar tamaño de etiquetas
            plt.tight_layout()

            # Guardar el gráfico en un archivo si se especifica una ruta
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")

        except Exception as e:
            print(f"Error en dot_plot: {e}")

    @staticmethod
    def bar_plot(df: pd.DataFrame, output_file: str = None):
        """
        Genera un gráfico de barras (Bar Plot) para destacar los términos más enriquecidos.

        :param df: DataFrame con columnas ['Term', 'Adjusted P-value'].
        :param output_file: Ruta para guardar el gráfico (opcional).
        """
        try:
            # Ordenar los términos por significancia (los más significativos primero)
            df = df.sort_values("Adjusted P-value", ascending=True).head(20)

            # Crear el gráfico de barras
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=df,
                x="Adjusted P-value",
                y="Term",
                hue="Term",  # Asignar hue al mismo eje y para evitar la advertencia
                dodge=False,  # Evitar separación innecesaria de barras
                errorbar=None,  # Reemplazar ci=None con errorbar=None
                palette="Spectral",  # Paleta de colores más atractiva
                legend=False,  # No mostrar la leyenda automática generada por hue
            )
            plt.xscale("log")  # Escala logarítmica para resaltar diferencias
            plt.title(
                "Bar Plot - Enrichment Analysis", fontsize=16, fontweight="bold"
            )  # Título con estilo
            plt.xlabel(
                "Adjusted P-value (log scale)", fontsize=14
            )  # Etiqueta X mejorada
            plt.ylabel("Term", fontsize=14)  # Etiqueta Y mejorada
            plt.xticks(fontsize=12)  # Ajustar tamaño de etiquetas en eje X
            plt.yticks(fontsize=12)  # Ajustar tamaño de etiquetas en eje Y
            plt.grid(
                axis="x", linestyle="--", alpha=0.7
            )  # Agregar líneas de referencia

            # Guardar el gráfico en un archivo si se especifica una ruta
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                print(f"Gráfico guardado en {output_file}")

        except Exception as e:
            print(f"Error en bar_plot: {e}")

    @staticmethod
    def cnet_plot_igraph(
        df: pd.DataFrame, gene_sets: Dict[str, List[str]], output_file: str = None
    ):
        """
        Genera un gráfico de red (Cnetplot) mostrando las relaciones entre genes y términos enriquecidos usando igraph.

        :param df: DataFrame con columnas ['Term', 'Genes'].
        :param gene_sets: Diccionario donde las claves son términos enriquecidos y los valores son listas de genes.
        :param output_file: Ruta para guardar el gráfico (opcional).
        """
        try:
            # Initialize an iGraph object
            graph = ig.Graph()

            # Add nodes for terms and genes, avoiding duplicates
            term_nodes = [
                {"name": term, "type": "term", "size": 20} for term in gene_sets.keys()
            ]
            gene_nodes = [
                {
                    "name": gene.replace("[", "").replace("]", "").replace('"', ""),
                    "type": "gene",
                    "size": 10,
                }
                for genes in gene_sets.values()
                for gene in genes
            ]

            # Combine nodes and remove duplicates by "name"
            all_nodes = {node["name"]: node for node in term_nodes + gene_nodes}
            graph.add_vertices([node["name"] for node in all_nodes.values()])

            # Add edges between terms and genes
            edges = []
            for term, genes in gene_sets.items():
                for gene in genes:
                    cleaned_gene = (
                        gene.replace("[", "").replace("]", "").replace('"', "")
                    )
                    edges.append((term, cleaned_gene))
            graph.add_edges(edges)

            # Set node attributes (size and type)
            graph.vs["size"] = [all_nodes[v["name"]]["size"] for v in graph.vs]
            graph.vs["type"] = [all_nodes[v["name"]]["type"] for v in graph.vs]

            # Define color mapping: terms -> blue, genes -> red
            graph.vs["color"] = [
                "skyblue" if v["type"] == "term" else "salmon" for v in graph.vs
            ]

            # Modify term labels: name in two rows (split by '\n')
            graph.vs["label"] = [
                v["name"].replace("(", "\n(") if v["type"] == "term" else v["name"]
                for v in graph.vs
            ]

            # Generate a layout
            layout = graph.layout("fruchterman_reingold")

            graph.vs["frame_color"]=  graph.vs['color']

            # Plot the graph
            plt.figure(figsize=(12, 10))
            ig.plot(
                graph,
                target=plt.gca(),
                layout=layout,
                vertex_size=graph.vs["size"],
                vertex_color=graph.vs["color"],
                vertex_label=graph.vs["label"],
                vertex_label_size=8,
                vertex_label_dist=2,
                edge_width=0.5,
                edge_color="gray",
                frame_color=graph.vs["frame_color"]
            )

            # Add a legend
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="skyblue",
                    markersize=10,
                    label="Terms",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="salmon",
                    markersize=10,
                    label="Genes",
                ),
            ]
            plt.legend(
                handles=legend_elements, loc="upper left", fontsize=10, frameon=True
            )

            # Add title
            plt.title("Cnetplot - Gene-Term Relationships (iGraph)", fontsize=16)
            plt.tight_layout()

            # Save the output
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")
            else:
                plt.show()

        except Exception as e:
            print(f"Error en cnet_plot: {e}")

    @staticmethod
    def cnet_plot(
        df: pd.DataFrame, gene_sets: Dict[str, List[str]], output_file: str = None
    ):
        """
        Genera un gráfico de red (Cnetplot) mostrando las relaciones entre genes y términos enriquecidos.

        :param df: DataFrame con columnas ['Term', 'Genes'].
        :param gene_sets: Diccionario donde las claves son términos enriquecidos y los valores son listas de genes.
        :param output_file: Ruta para guardar el gráfico (opcional).
        """
        try:
            # Crear el grafo
            G = nx.Graph()

            # Agregar nodos para términos enriquecidos y genes
            for term, genes in gene_sets.items():
                G.add_node(term, type="term", size=20)
                for gene in genes:
                    G.add_node(gene, type="gene", size=10)
                    G.add_edge(term, gene)

            # Configurar la posición de los nodos para una distribución uniforme
            pos = nx.spring_layout(
                G, seed=42, k=0.8
            )  # k intermedio para balancear repulsión y atracción

            # Normalizar las posiciones para cubrir uniformemente el área del gráfico
            pos = {node: (x * 10, y * 10) for node, (x, y) in pos.items()}

            # Dibujar nodos y aristas con estilos mejorados
            plt.figure(figsize=(14, 12))
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[n for n, d in G.nodes(data=True) if d["type"] == "term"],
                node_size=800,
                node_color="skyblue",
                edgecolors="black",
                linewidths=1.5,
                label="Terms",
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[n for n, d in G.nodes(data=True) if d["type"] == "gene"],
                node_size=400,
                node_color="salmon",
                edgecolors="black",
                linewidths=1.5,
                label="Genes",
            )
            nx.draw_networkx_edges(G, pos, alpha=0.5)

            # Mostrar etiquetas para todos los nodos (términos y genes)
            nx.draw_networkx_labels(
                G,
                pos,
                labels={
                    n: n for n in G.nodes()
                },  # Mostrar etiquetas para todos los nodos
                font_size=8,
                font_color="darkblue",
            )

            # Agregar título y leyenda mejorada
            plt.title("Cnetplot - Gene-Term Relationships", fontsize=16)
            plt.legend(frameon=True, loc="upper left", fontsize=10)
            plt.tight_layout()

            # Guardar el gráfico en un archivo si se especifica una ruta
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")

        except Exception as e:
            print(f"Error en cnet_plot: {e}")

    @staticmethod
    def upset_plot(df: pd.DataFrame, output_file: Optional[str] = None):
        """
        Genera un gráfico UpSet que muestra las intersecciones entre términos GO y clusters.

        Este gráfico es útil para analizar la distribución de términos en múltiples clusters,
        identificando solapamientos o patrones específicos.

        :param df:
            DataFrame que contiene los datos para generar el gráfico. Debe incluir las siguientes columnas:
                - 'Term': Nombre de los términos GO analizados.
                - 'Cluster': Identificador del cluster al que pertenece cada término.
        :param output_file:
            Ruta opcional donde se guardará el gráfico generado como archivo PDF. Si no se proporciona,
            el gráfico se mostrará en pantalla.

        :return:
            None. La función muestra o guarda el gráfico generado.
        """
        try:
            # Verificar que las columnas necesarias estén presentes
            if not {"Term", "Cluster"}.issubset(df.columns):
                raise ValueError(
                    "El DataFrame debe contener las columnas 'Term' y 'Cluster'."
                )

            # Seleccionar solo las columnas requeridas
            filtered_data = df[["Term", "Cluster"]]

            # Agrupar por 'Cluster' y 'Term', contar ocurrencias y reorganizar en una matriz binaria
            binary_data = (
                filtered_data.groupby(["Cluster", "Term"])
                .size()  # Contar las ocurrencias de cada par (Cluster, Term)
                .unstack(
                    fill_value=0
                )  # Expandir 'Term' en columnas, rellenar valores ausentes con 0
                .astype(bool)  # Convertir las cuentas a valores booleanos
            )

            # Restablecer el índice para convertir los clusters en una columna
            binary_data = binary_data.reset_index()

            # Renombrar la columna del índice de cluster como 'count'
            # Esto es necesario porque UpSet utiliza 'count' como marcador de tamaños de subconjunto
            binary_data = binary_data.rename(columns={"Cluster": "count"})

            # Seleccionar columnas que no sean 'count' para establecer el índice
            index_columns = [col for col in binary_data.columns if col != "count"]
            binary_data = binary_data.set_index(index_columns)

            # Crear el gráfico UpSet
            upset_plot = UpSet(
                binary_data,
                subset_size="count",  # Especifica que el tamaño de subconjuntos se basa en la columna 'count'
                show_counts=True,  # Muestra conteos en las barras del gráfico
            )

            # Generar el gráfico
            upset_plot.plot()

            # Añadir un título descriptivo al gráfico
            plt.suptitle("UpSet Plot de Términos GO y Clusters")

            # Guardar el gráfico si se proporciona una ruta de archivo
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")

        except Exception as e:
            print(f"Error en upset_plot: {e}")

    @staticmethod
    def venn_diagram(
        file_modularity: str, file_enrichment: str, output_file: Optional[str] = None
    ):
        """
        Genera un diagrama de Venn comparando dos conjuntos de términos enriquecidos de dos métodos de clustering
        (Leiden  con Modularidad Máxima y Leiden con Máximo Puntuaje de Enrequecimiento Funcional ),
        donde los términos se extraen de archivos CSV correspondientes a cada resultado."

        :param file1: Ruta del primer archivo CSV que contiene una columna 'Term'.
        :param file2: Ruta del segundo archivo CSV que contiene una columna 'Term'.
        :param output_file: Ruta opcional para guardar el gráfico generado (en formato PNG).
        """

        try:
            # Leer archivos CSV
            data_leiden_max_modularity = pd.read_csv(file_modularity)
            data_leiden_max_enrichment = pd.read_csv(file_enrichment)

            # Extraer términos únicos del primer archivo
            terms_modularity = set(data_leiden_max_modularity["Term"])
            terms_enrichment = set(data_leiden_max_enrichment["Term"])

            # Obtener la longitud de cada conjunto (opcional para análisis interno)
            len(terms_modularity), len(terms_enrichment)

            # Crear el diagrama de Venn comparando los dos conjuntos
            venn = venn2(
                [terms_modularity, terms_enrichment],
                (
                    f"Leiden (Máxima Modularidad)\nTotal: {len(terms_modularity)}",
                    f"Leiden (Máximo Puntuaje de Enrequecimiento Funcional)\nTotal: {len(terms_enrichment)}",
                ),
            )

            # Añadir un título al diagrama
            plt.title(
                "Comparación de términos funcionales entre las soluciones \n"
                "con el algoritmo de Leiden (Máxima Modularidad vs Máxima Significancia Biológica)"
            )

            # Guardar el gráfico en un archivo si se especifica una ruta
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")

        except Exception as e:
            print(f"Error en venn_diagram_from_csv: {e}")
