import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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
    def bar_plot(df: pd.DataFrame):
        try:
            df = df.sort_values('Adjusted P-value').head(10)
            plt.barh(df['Term'], df['Adjusted P-value'], color='skyblue')
            plt.title('Bar Plot')
            plt.xlabel('Adjusted P-value')
            plt.ylabel('Term')
            plt.show()
        except:
            print("Error in bar_plot")

    @staticmethod
    def cnet_plot(gene_sets):
        try:
            G = nx.Graph()
            for term, genes in gene_sets.items():
                G.add_node(term)  # Nodo para el término
                for gene in genes:
                    G.add_node(gene)  # Nodo para cada gen
                    G.add_edge(term, gene)  # Enlace entre término y gen

            # Layout básico
            pos = nx.spring_layout(G)
            
            # Dibujar nodos y aristas
            nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8)
            plt.title('Cnetplot')
            plt.show()
        except:
            print("Error en cnet_plot")
            

