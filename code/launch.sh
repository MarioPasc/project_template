#! /usr/bin/env bash

# variables de entorno R_LIB, PYTHON_LIB
export PYTHON_LIB=./Py_libs

# DESCARGA DE DATOS
mkdir -p data
# obtener genes de HPO
curl -X 'GET' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -H 'accept: application/json' -o data/hpoGenes.tsv -s
awk 'NR > 1{print $2}' data/hpoGenes.tsv > data/hpoGenesNames.tsv # obtener nombres de los genes

#obtener red de String db
./obtainPPINetwork.py data/hpoGenesNames.tsv #--filter   --nodes

