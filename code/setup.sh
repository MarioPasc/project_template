#! /usr/bin/env bash

export PYTHON_LIB=./Py_libs
mkdir -p $PYTHON_LIB

pip install --target=$PYTHON_LIB pandas
pip install --target=$PYTHON_LIB stringdb


