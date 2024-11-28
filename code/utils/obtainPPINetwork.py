#!/usr/bin/env python3
import argparse
import stringdb
from typing import Union, List
import os
import logging

# Create the logs directory
log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok=True)

# Configure logging of this stage of the project
logging.basicConfig(
    filename=os.path.join(log_folder, "data_downloading_logging.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def obtain_genes(file_path: Union[str, os.PathLike]) -> List[str]:
    """
    Obtains a list of genes from a file where each gene is expected to be on a new line.

    Args
    --------------------------
    file_path (Union[str, os.PathLike]): The path to the file containing the list of genes.

    Returns
    --------------------------
    gene_list (List[str]): A list of gene names (strings) extracted from the file.
    """
    genes_list: List[str] = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():  # Avoid adding empty lines
                    genes_list.append(line.strip())
        logging.info(f"Successfully read {len(genes_list)} genes from {file_path}")

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the gene file: {e}")
        raise

    return genes_list


def main():
    """
    Obtains an interaction network from a gene list using the StringDB API.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Obtain the interaction network from StringDB using a gene list."
    )
    parser.add_argument(
        "gene_file",
        type=str,
        help="Path to the file containing the list of genes (e.g., data/genes.tsv)",
    )
    parser.add_argument(
        "network_path", type=str, help="Path where the network file will be written"
    )

    parser.add_argument(
        "-f",
        "--filter",
        type=int,
        default=400,
        help="Combine score threshold for filtering the network (range: 0-1000, default: 400)",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        default=0,
        help="Number of additional nodes to add to the network (must be non-negative, default: 0)",
    )

    args = parser.parse_args()

    # Validate and handle invalid arguments
    if not os.path.isfile(args.gene_file):
        logging.error(f"The gene file '{args.gene_file}' does not exist.")
        print(f"Error: The gene file '{args.gene_file}' does not exist.")
        return

    if not (0 <= args.filter <= 1000):
        logging.warning(
            f"Invalid filter value: {args.filter}. Using the default value of 400."
        )
        args.filter = 400

    if args.nodes < 0:
        logging.warning(
            f"Invalid number of nodes: {args.nodes}. Using the default value of 0."
        )
        args.nodes = 0

    # Obtain the gene list
    try:
        genes_list = obtain_genes(args.gene_file)
    except Exception as e:
        logging.error(f"Failed to obtain the gene list: {e}")
        return

    if not genes_list:
        logging.error("The gene list is empty or could not be read.")
        print(
            "Error: The gene list is empty or could not be read. Please check the input file."
        )
        return

    # Obtain STRING IDs from the genes
    try:
        string_ids = stringdb.get_string_ids(
            genes_list
        )  # Default species is Homo sapiens (9606)
        logging.info(f"Successfully obtained STRING IDs for {len(genes_list)} genes.")
    except Exception as e:
        logging.error(f"Failed to obtain STRING IDs: {e}")
        print(f"Error: Failed to obtain STRING IDs for the given genes. Details: {e}")
        return

    if string_ids.empty:
        logging.error("No STRING IDs were retrieved. Check the provided gene names.")
        print(
            "Error: No STRING IDs were retrieved. Please check if the provided gene names are correct."
        )
        return

    # Obtain the interaction network
    try:
        network = stringdb.get_network(
            string_ids["stringId"], required_score=args.filter, add_nodes=args.nodes
        )
        network.to_csv(args.network_path, sep="\t")
        logging.info(
            "The interaction network was successfully saved to 'data/network.tsv'."
        )
    except Exception as e:
        logging.error(f"Failed to obtain the interaction network: {e}")
        print(f"Error: Failed to obtain the interaction network. Details: {e}")
        return


if __name__ == "__main__":
    main()
