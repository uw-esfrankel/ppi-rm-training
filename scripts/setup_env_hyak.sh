#!/bin/bash

# Setup environment for PPI-RM training
# This script installs required dependencies and sets up the environment

# if uv is not installed, install it, otherwise just say uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 
else
    echo "uv is installed"
fi

module load cuda/12.4.1 
module load gcc/9.3.0

# install uv environment if not already installed, otherwise just say uv environment is installed
if [ ! -d "dataset/.venv" ]; then
    echo "uv environment not found. Installing uv environment..."
    cd dataset
    uv venv --python 3.10
    uv pip install -r requirements.txt
    cd ..
else
    echo "uv environment already exists"
fi

