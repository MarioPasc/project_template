#! /usr/bin/env bash

export PYTHON_LIB=./Py_libs
mkdir -p $PYTHON_LIB

pip install --target=$PYTHON_LIB numpy
pip install --target=$PYTHON_LIB pandas
pip install --target=$PYTHON_LIB stringdb
pip install --target=$PYTHON_LIB igraph
pip install --target=$PYTHON_LIB matplotlib
pip install --target=$PYTHON_LIB cairocffi
pip install --target=$PYTHON_LIB plotly
pip install --target=$PYTHON_LIB statistics
pip install --target=$PYTHON_LIB jinja2
pip install --target=$PYTHON_LIB networkx
pip install --target=$PYTHON_LIB optuna
pip install --target=$PYTHON_LIB requests
pip install --target=$PYTHON_LIB psutil
pip install --target=$PYTHON_LIB scienceplots
pip install --target=$PYTHON_LIB kaleido
