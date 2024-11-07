#!/usr/bin/env python3 
import argparse
import stringdb
import pandas as pd


def obtain_genes(file_path):
    """
    Description
    Obtains a list of genes from a file where each gene is expected to be on a new line.
    --------------------------
    Args
    file_path: str
    The path to the file containing the list of genes.
    --------------------------
    Returns
    gene_list: list
    A list of gene names (strings) extracted from the file.
    --------------------------
    Throws
    Exception
    A generic catch-all.
    --------------------------
    """
    genes_list=[]
    try:
        with open(file_path, 'r') as f:
            for line in f:
                genes_list.append(line.strip())
    except Exception as e:
        print(f"Se ha producido un error: {e}")

    return genes_list


def main():
    """
    Obtains from a gene list a interaction network using stringdb API.
    """
    parser = argparse.ArgumentParser(description='Obtener la red de interacción de StringDB a partir de una lista de genes')
    parser.add_argument('gene_file', type=str, help='Ruta del archivo de genes (ejemplo: data/genes.tsv)')
    parser.add_argument('-f',"--filter", type=int, default=400, help='Valor de combine score de la red, por el que será filtrada (optional)')
    parser.add_argument('-n',"--nodes", type=int, default=0, help='Número de nodos a añadir a la red (optional)')

    args = parser.parse_args()

    genes_list= obtain_genes(args.gene_file)

    #obtener string ids a partir de los genes
    string_ids= stringdb.get_string_ids(genes_list) #especie por defecto homo sapiens (9606)
    #string_ids.to_csv("data/stringID.csv")


    if not (0 <= args.filter <= 1000): 
        print(f"Valor de filtrado {args.filter}, no válido. Se ha empleado el valor por defecto")
        args.filter=400

    if args.nodes < 0:  
        print(f"El número de nodos a añadir {args.nodes}, no es correcto. No se han añadido nodos")
        args.nodes=0


    # obtener la red de interacción
    network= stringdb.get_network(string_ids["stringId"], required_score=args.filter, add_nodes=args.nodes)
    network.to_csv("data/network.tsv", sep="\t")


if __name__ == "__main__":
    main()



