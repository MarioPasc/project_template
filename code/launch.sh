#! /usr/bin/env bash

# Path variables of Python and R
export PYTHON_LIB=./Py_libs
export PYTHONPATH=$PYTHON_LIB:$PYTHONPATH  # Añadir Py_libs al PYTHONPATH

# Create the data directory if it doesn't exist
mkdir -p data
# obtener genes de HPO
# ES NECESARIO INSTALAR CURL YA QUE NO VIENE INSTALADO POR DEFECTO
curl -X 'GET' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -H 'accept: application/json' -o data/hpoGenes.tsv -s
hpo_genes="data/hpoGenesNames.tsv"
awk 'NR > 1{print $2}' data/hpoGenes.tsv > $hpo_genes  # Extract gene names

# Proceed with the rest of your script
network="data/network.tsv"
./obtainPPINetwork.py $hpo_genes $network
network/networkAnalysis.py $network