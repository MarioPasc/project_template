import pandas as pd
import matplotlib.pyplot as plt

class FunctionalVisualization:
    @staticmethod
    def dot_plot(df: pd.DataFrame):
        try:
            df = df.sort_values('Adjusted P-value').head(10)
            plt.scatter(df['Gene Ratio'], df['Term'], s=100, c=df['Adjusted P-value'], cmap='coolwarm')
            plt.title('Dot Plot')
            plt.xlabel('Gene Ratio')
            plt.ylabel('Term')
            plt.show()
        except:
            print("Error in dot_plot")

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
