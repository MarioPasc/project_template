#!/usr/bin/env python3

# TODO: Gonzalo, implementa estos 3 algoritmos: 
#  - Walktrap: 
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_walktrap. 
#       Se debe poder ajustar el parámetro "steps"
#       Referencia: 
#           PONS, Pascal; LATAPY, Matthieu. 
#           Computing communities in large networks using random walks. 
#           En Computer and Information Sciences-ISCIS 2005: 20th International Symposium, 
#           Istanbul, Turkey, October 26-28, 2005. Proceedings 20. Springer Berlin Heidelberg, 2005. p. 284-293.
#
#   - Leiden:
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_leiden
#       Se debe poder ajustar "resolution". Fijaremos n_iterations a -1 y objective_function a "modularity"
#       Referencia:
#           TRAAG, Vincent A.; WALTMAN, Ludo; VAN ECK, Nees Jan. 
#           From Louvain to Leiden: guaranteeing well-connected communities. 
#           Scientific reports, 2019, vol. 9, no 1, p. 1-12.
#
#   - Louvain:
#       https://python.igraph.org/en/stable/api/igraph.Graph.html#community_multilevel
#       Se debe poder ajustar "resolution". Fijaremos return_levels = False, en un principio. 
#       Referencia:
#           BLONDEL, Vincent D., et al. 
#           Fast unfolding of communities in large networks. 
#           Journal of statistical mechanics: theory and experiment, 2008, vol. 2008, no 10, p. P10008.

# NOTE: Ten en cuenta que estos algoritmos devuelven un objeto tipo VertexClustering (https://python.igraph.org/en/stable/api/igraph.VertexClustering.html)
#       cuando programe la optimización de hiperparámetros, yo no voy a preocuparme de pasar de VertexClustering al formato que esperan las métricas de la 
#       clase Metrics, así que sería genial si pudieras adaptar las funciones de metrics.py para que funcionen con VertexClustering, o adaptar la salida de
#       los algoritmos para que devuelvan algún otro formato.  

class Algorithms:
    def clustering():
        return 0