#! /usr/bin/env bash

# variables de entorno R_LIB, PYTHON_LIB
export PYTHON_LIB=./Py_libs

# DESCARGA DE DATOS
mkdir -p data
# obtener genes de HPO
curl -X 'GET' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -H 'accept: application/json' -o data/hpoGenes.tsv -s
hpo_genes="data/hpoGenesNames.tsv"
awk 'NR > 1{print $2}' data/hpoGenes.tsv > $hpo_genes # obtener nombres de los genes

#obtener red de String db
network="data/network.tsv"
./obtainPPINetwork.py $hpo_genes  $network #--filter   --nodes
./networkAnalysis.py $network

#?puedo hacer que me devuelva el nombre del fichero de network

