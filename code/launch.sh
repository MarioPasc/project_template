#!/usr/bin/env bash

# Variables por defecto
export RESULTS="./results"
export CONTAINER_NAME="biosist_ftd_run"

# Permitir al usuario definir OPTIMIZE y TRIALS al momento de la ejecución
export OPTIMIZE=${OPTIMIZE:-false}
export TRIALS=${TRIALS:-100}

# Asegurar que las carpetas existan
mkdir -p "$(pwd)/code/data"
mkdir -p "$(pwd)/results"

# Ejecutar Docker con montaje de volúmenes y variables de entorno
docker run --rm \
-v "$(pwd)/results:/app/results" \
-v "$(pwd)/code/data:/app/code/data" \
-v "$(pwd)/report:/app/report" \
-e OPTIMIZE="$OPTIMIZE" \
-e TRIALS="$TRIALS" \
--name "$CONTAINER_NAME" \
mpascualg/biosist_ftd:latest

