#! /usr/bin/env bash

# Set to true if you want to fine-tune the clustering algorithms. 
train=false
# If train is true, then set the number of trials to run the Bayesian Optimization Algorithm. 
# In our original study we executed it for 150 trials.
trials=150

# Path to look for the libraries of Python
export PYTHON_LIB=./Py_libs
export PYTHONPATH=$PYTHON_LIB:./code:$PYTHONPATH  # Include both Py_libs and ./code in PYTHONPATH


# Create the data directory if it doesn't exist
mkdir -p ./code/data

# Download the HPO genes file using wget instead of curl, which is not installed by default in most Linux distros
hpo_genes_data="./code/data/hpoGenes.tsv"

wget -q --header='Accept: application/json' 'https://ontology.jax.org/api/network/annotation/HP%3A0002145/download/gene' -O $hpo_genes_data

hpo_genes="./code/data/hpoGenesNames.tsv"

awk 'NR > 1{print $2}' $hpo_genes_data > $hpo_genes  # Extract gene names

# Proceed with the rest of your script
network="./code/data/network.tsv"
results="./results"

# Retrieve the Protein-Protein Interaction Network
./code/utils/obtainPPINetwork.py $hpo_genes $network --filter="700" 

# Perform Network Analysis
./code/network/analysis.py $network $results

# Run optimization only if 'train' is true
if [ "$train" = true ]; then
    echo "Running optimization..."

    # Optimize Louvain
    ./code/clustering/optimize.py \
        --config_path code/clustering/configs/multilevel.yaml \
        --network_csv code/data/network.tsv \
        --study_name multilevel_optimization \
        --output_path results \
        --n_trials $trials

    # Optimize Leiden
    ./code/clustering/optimize.py \
        --config_path code/clustering/configs/leiden.yaml \
        --network_csv code/data/network.tsv \
        --study_name leiden_optimization \
        --output_path results \
        --n_trials $trials
fi

# Store optimization results

./code/clustering/bho_analysis.py \
    results/

# Store clustering results
./code/clustering/analysis.py \
    code/data/network.tsv \
    results/