#!/usr/bin/env python3

import logging
import igraph
import os
import pandas as pd
import logging
from typing import Union

# en clustering esta también
def network_to_igraph_format(network_csv: Union[str, os.PathLike], sep: str ="\t") -> igraph.Graph:
    """
    Converts a network, in a file, to an igraph format.

    Args:
        network_csv (Union[str, os.PathLike]): File path to the network.
        sep (str): Delimiter of the file (default: "\t").

    Returns:
        igraph.Graph: Graph in an igraph format, or None if an error occurs.

    """
    try:
        network_df= pd.read_csv(network_csv, sep=sep, header=0) #cambiar separador si poneis otro formato que no sea tsv
        graph= igraph.Graph.DataFrame(network_df[['preferredName_A', 'preferredName_B']], directed=False, use_vids=False)
        logging.info(f"Successfully converts network to igraph format")
        return graph
    except FileNotFoundError:
        logging.error(f"File not found: {network_csv}")
        raise
    except pd.errors.ParserError:
        logging.error(f"File not in the right format:  {network_csv}")
        raise
    except KeyError:
        logging.error(f"Columns 'preferredName_A' and 'preferredName_B' not file: {network_csv} ")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

