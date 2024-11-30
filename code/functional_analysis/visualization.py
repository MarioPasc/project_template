import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict

def prepare_data_for_visualization_from_df(df: pd.DataFrame):
    try:
        # Convertir Overlap a Gene Ratio (proporción)
        df[['Observed', 'Total']] = df['Overlap'].str.split('/', expand=True).astype(int)
        df['Gene Ratio'] = df['Observed'] / df['Total']

        # Crear un diccionario para cnetplot
        gene_sets = {
            row['Term']: row['Genes'].split(', ') for _, row in df.iterrows()
        }

        return df, gene_sets

    except Exception as e:
        print(f"Error en prepare_data_for_visualization_from_df: {e}")
        return None, None

class FunctionalVisualization:
    @staticmethod
    def dot_plot(df: pd.DataFrame, output_file: str = None):
        try:
            # Ordenar los términos de significancia (los más significativos primero)
            df = df.sort_values('Adjusted P-value', ascending=True).head(20)
            
            # Crear un gráfico de puntos
            plt.figure(figsize=(10, 8))
            scatter = sns.scatterplot(
                data=df,
                x='Gene Ratio',
                y='Term',
                size='Adjusted P-value',
                hue='Adjusted P-value',
                sizes=(50, 300),  # Aumentar el rango de tamaños de los puntos
                palette='coolwarm',  # Mejorar contraste de colores
                legend='brief'
            )
            plt.title('Dot Plot - Enrichment Analysis', fontsize=14)
            plt.xlabel('Gene Ratio', fontsize=12)
            plt.ylabel('Term', fontsize=12)
            plt.gca().yaxis.set_tick_params(labelsize=10)  # Mejorar tamaño de etiquetas
            plt.tight_layout()
            
            # Guardar o mostrar el gráfico
            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"Gráfico guardado en {output_file}")
            plt.show()
            
        except Exception as e:
            print(f"Error en dot_plot: {e}")

    @staticmethod
    def bar_plot(df: pd.DataFrame, output_file:str = None):
        try:
            # Ordenar los términos por significancia (los más significativos primero)
            df = df.sort_values('Adjusted P-value', ascending=True).head(20)
            
            # Crear el gráfico de barras
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=df,
                x='Adjusted P-value',
                y='Term',
                hue='Term',  # Resolver FutureWarning
                dodge=False,  # Prevenir separación innecesaria
                palette='Spectral'  # Paleta más contrastante
            )
            plt.xscale('log')  # Usar escala logarítmica para resaltar diferencias
            plt.title('Bar Plot - Enrichment Analysis', fontsize=14)
            plt.xlabel('Adjusted P-value (log scale)', fontsize=12)
            plt.ylabel('Term', fontsize=12)
            plt.gca().yaxis.set_tick_params(labelsize=10)  # Tamaño de etiquetas
            plt.tight_layout()
            
            # Guardar o mostrar el gráfico
            if output_file:
                plt.savefig(output_file, dpi= 300)
                print(f"Gráfico guardado en {output_file}")
            plt.show()
            
        except Exception as e:
            print(f"Error en bar_plot: {e}")

    @staticmethod
    def cnet_plot(gene_sets):
        try:
            # Crear el grafo
            G = nx.Graph()
            
            # Agregar nodos para términos enriquecidos y genes
            for term, genes in gene_sets.items():
                G.add_node(term, type='term', size=10)       # Nodo para el término
                for gene in genes:
                    G.add_node(gene, type='gene', size=10)   # Nodo para cada gen
                    G.add_edge(term, gene)                   # Enlace entre término y gen

            # Configurar la posición de los nodos con menor espaciado
            pos = nx.spring_layout(G, seed=42, k=0.5)  # Reducir 'k' para compactar el gráfico

            # Dibujar los nodos y las aristas
            plt.figure(figsize=(12, 10))  # Ajustar el tamaño del gráfico
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'term'],
                node_size=700, node_color='lightblue', label='Terms'
            )
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'gene'],
                node_size=300, node_color='orange', label='Genes'
            )
            nx.draw_networkx_edges(G, pos, alpha=0.5)

            # Mostrar etiquetas para todos los nodos (términos y genes)
            nx.draw_networkx_labels(
                G, pos,
                labels={n: n for n in G.nodes()},
                font_size=8  # Ajustar tamaño de la fuente
            )

            # Agregar título y leyenda
            plt.title('Cnetplot - Gene-Term Relationships', fontsize=14)
            plt.legend()
            plt.tight_layout()

            # Guardar o mostrar el gráfico
            if output_file:
                plt.savefig(output_file)
                print(f"Gráfico guardado en {output_file}")
            plt.show()

        except Exception as e:
            print(f"Error en cnet_plot: {e}")
            

