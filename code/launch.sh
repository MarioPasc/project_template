#! /usr/bin/env bash

# Set to true if you want to fine-tune the clustering algorithms.
export OPTIMIZE=false

# If train is true, then set the number of trials to run the Bayesian Optimization Algorithm.
# In our original study we executed it for 150 trials.
export TRIALS=100

# Results folder
export RESULTS="./results"

# Verbose for execution. We highly recommend setting this to true. This verbose is minimum,
# all the details are saved in the logs (./logs/{name}.log)
export VERBOSE=1

# Path to look for the libraries of Python
export PYTHON_LIB=./Py_libs
export PYTHONPATH=$PYTHON_LIB:./code:$PYTHONPATH  # Include both Py_libs and ./code in PYTHONPATH

# Print sesion info
./code/utils/sesion_info.py

# Create the data directory if it doesn't exist
mkdir -p ./code/data

# Download the HPO genes file using wget instead of curl, which is not installed by default in most Linux distros
hpo_genes_data="./code/data/hpoGenes.tsv"

#wget -q --header='Accept: application/json' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -O $hpo_genes_data

hpo_genes="./code/data/hpoGenesNames.tsv"

awk 'NR > 1{print $2}' $hpo_genes_data > $hpo_genes  # Extract gene names

# Proceed with the rest of your script
network="./code/data/network.tsv"

# Retrieve the Protein-Protein Interaction Network
#./code/utils/obtainPPINetwork.py $hpo_genes $network --filter="700"

# Perform Network Analysis
#./code/network/analysis.py $network $RESULTS

# Run optimization only if 'train' is true
if [ $OPTIMIZE = true ]; then
    # Optimize Louvain
    ./code/clustering/optimize.py \
    --config_path code/clustering/configs/multilevel.yaml \
    --network_csv code/data/network.tsv \
    --study_name multilevel_optimization \
    --output_path $RESULTS \
    --n_trials $TRIALS
    
    # Optimize Leiden
    ./code/clustering/optimize.py \
    --config_path code/clustering/configs/leiden.yaml \
    --network_csv code/data/network.tsv \
    --study_name leiden_optimization \
    --output_path $RESULTS \
    --n_trials $TRIALS
    
fi

# Performance analysis on optimization results
#./code/clustering/bho_analysis.py $RESULTS

# Clustering visualization and final results saving
#./code/clustering/analysis.py code/data/network.tsv $RESULTS

./code/functional_analysis/complete_analysis.py $network $RESULTS $RESULTS/clustering_results.json -f pdf -a leiden_max_enrichment -p 0.005 -c 2000 -o 0.1