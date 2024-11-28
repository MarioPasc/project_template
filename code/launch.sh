#! /usr/bin/env bash

# Path to look for the libraries of Python
export PYTHON_LIB=./Py_libs
export PYTHONPATH=$PYTHON_LIB:./code:$PYTHONPATH  # Include both Py_libs and ./code in PYTHONPATH


# Create the data directory if it doesn't exist
mkdir -p ./code/data

# Download the HPO genes file using wget instead of curl, which is not installed by default in most Linux distros
wget -q --header='Accept: application/json' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -O data/hpoGenes.tsv

hpo_genes="./code/data/hpoGenesNames.tsv"

awk 'NR > 1{print $2}' data/hpoGenes.tsv > $hpo_genes  # Extract gene names

# Proceed with the rest of your script
network="./code/data/network.tsv"

# Retrieve the Protein-Protein Interaction Network
./code/utils/obtainPPINetwork.py $hpo_genes $network --filter="800" 

# Perform Netowkr Analysis
./code/network/analysis.py $network