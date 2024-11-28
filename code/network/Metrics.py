import igraph 
import plotly.graph_objects as go
from typing import Tuple
import logging


class Metrics:
    @staticmethod
    def analysis_degree(graph: igraph.Graph)-> Tuple[float, float]:
        """
        Analyzes the degree distribution of a graph and saves a histogram in HTML format.

        Args:
            graph (igraph.Graph): The graph to analyze.

        Returns:
            tuple(float, float): Mean degree and standard deviation of the degree distribution.
        """
        try:
            if not isinstance(graph, igraph.Graph):
                raise ValueError("The provided graph is not an igraph.Graph object.")
            
            # degree of each node
            #degree_dict = dict(zip(graph.vs["name"], graph.degree()))

            #obtain the degree distribution of the graph
            degree_distribution= graph.degree_distribution()

            mean_degree=degree_distribution.mean
            sd_degree=degree_distribution.sd

            #histogram of the degree distribution using plotly
            bins = list(degree_distribution.bins()) # list of tuples (left_limit, right_limit, count)

            x = [bin[0] for bin in bins]  # left limit
            y = [bin[2] for bin in bins]  # counts

            fig = go.Figure(data=[go.Bar(x=x, y=y)])

            fig.update_layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Frecuency",
            bargap=0.2
            )

            #to include in the histogram the mean and std
            fig.add_annotation(x=34,  y=4,  
                text=f'Mean: {mean_degree:.2f}<br>Standar Deviation: {sd_degree:.2f}',  
                showarrow=False,  
                font=dict(size=14, color="black"),
                align="right",  
                bgcolor="white",
                opacity=0.8  
            )

            fig.show()
            fig.write_html("../results/degree_distribution.html")
            return mean_degree, sd_degree
        except ValueError as v:
            logging.error(f"ValueError: {v}")
            raise