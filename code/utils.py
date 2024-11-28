#!/usr/bin/env python3

import logging
import igraph
import os
import pandas as pd
from typing import Union

import os
import logging

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name, log file, and logging level.
    
    :param name: Name of the logger.
    :param log_file: Path to the log file.
    :param level: Logging level (default is DEBUG).
    :return: Configured logger.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

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
