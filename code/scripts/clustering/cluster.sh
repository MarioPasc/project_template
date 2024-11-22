#!/bin/bash

# Usage:
# ./run_clustering.sh config.yaml ./results 100 True

# Arguments
CONFIG_PATH="$1"
OUTPUT_PATH="$2"
N_TRIALS="${3:-100}"  # Default to 100 trials
SAVE_PLOTS="${4:-False}"  # Default to not save plots

# Execute Python script with arguments
python3 optimize_clustering.py \
    --config_path "$CONFIG_PATH" \
    --output_path "$OUTPUT_PATH" \
    --n_trials "$N_TRIALS" \
    --save_plots "$SAVE_PLOTS"
