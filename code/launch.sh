#! /usr/bin/env bash

# Path variables of Python and R
export PYTHON_LIB=./Py_libs
export PYTHONPATH=$PYTHON_LIB:$PYTHONPATH  

# Create the data directory if it doesn't exist
mkdir -p data

# Download the HPO genes file using wget instead of curl, which is not installed by default in most Linux distros
wget -q --header='Accept: application/json' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -O data/hpoGenes.tsv

# Process the genes file
hpo_genes="data/hpoGenesNames.tsv"
awk 'NR > 1{print $2}' data/hpoGenes.tsv > $hpo_genes  # Extract gene names

# Proceed with the rest of your script
network="data/network.tsv"
./obtainPPINetwork.py $hpo_genes $network
./networkAnalysis.py $network