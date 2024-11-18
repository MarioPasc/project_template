#!/usr/bin/env python3

import logging
import igraph
import os
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
def network_to_igraph_format(network_csv: Union[str, os.PathLike]) -> igraph.Graph:
    return None