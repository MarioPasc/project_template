#!/usr/bin/env bash

# Variables
export RESULTS="./results"
export CONTAINER_NAME="project_template_run"

# Asegurar que las carpetas existan
mkdir -p "$(pwd)/code/data"
mkdir -p "$(pwd)/results"

# Ejecutar Docker con montaje de volúmenes
docker run --rm \
-v "$(pwd)/results:/app/results" \
-v "$(pwd)/code/data:/app/code/data" \
-v "$(pwd)/code:/app/code" \
--name "$CONTAINER_NAME" \
project_template:latest