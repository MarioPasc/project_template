#! /usr/bin/env bash

export PYTHON_LIB=./Py_libs
mkdir -p $PYTHON_LIB

pip install --target=$PYTHON_LIB numpy --upgrade
pip install --target=$PYTHON_LIB pandas --upgrade
pip install --target=$PYTHON_LIB stringdb --upgrade
pip install --target=$PYTHON_LIB igraph --upgrade
pip install --target=$PYTHON_LIB matplotlib --upgrade
pip install --target=$PYTHON_LIB cairocffi --upgrade
pip install --target=$PYTHON_LIB plotly --upgrade
# pip install --target=$PYTHON_LIB statistics --upgrade
pip install --target=$PYTHON_LIB jinja2 --upgrade
pip install --target=$PYTHON_LIB networkx --upgrade
pip install --target=$PYTHON_LIB optuna --upgrade
pip install --target=$PYTHON_LIB requests --upgrade
pip install --target=$PYTHON_LIB psutil --upgrade
pip install --target=$PYTHON_LIB scienceplots --upgrade
pip install --target=$PYTHON_LIB kaleido --upgrade
pip install --target=$PYTHON_LIB gseapy --upgrade
pip install --target=$PYTHON_LIB seaborn --upgrade
pip install --target=$PYTHON_LIB UpSetPlot --upgrade
pip install --target=$PYTHON_LIB matplotlib-venn --upgrade