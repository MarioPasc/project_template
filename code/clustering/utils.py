#!/usr/bin/env python3

import logging
import igraph
import os
import pandas as pd
from typing import Union

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name, log file, and logging level.
    
    :param name: Name of the logger.
    :param log_file: Path to the log file.
    :param level: Logging level (default is DEBUG).
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler: logging.FileHandler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

# TODO: Gonzalo implementa esta función un besito
# lo he implementado yo (Carmen), porque también lo uso en mi parte
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
        return graph
    except FileNotFoundError:
        print(f"Error: The file {network_csv} could not be found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: The file {network_csv} is not in the right format.")
    except KeyError:
        print(f"Error: The columns 'preferredName_A' and 'preferredName_B' are not in the file ")
    except Exception as e:
        print(f"Unexpected error: {e}")

