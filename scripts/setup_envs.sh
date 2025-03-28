#!/bin/bash

# Setup environment for PPI-RM training
# This script installs required dependencies and sets up the environment

# if uv is not installed, install it, otherwise just say uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 
else
    echo "uv is installed"
fi

# install uv environment if not already installed, otherwise just say uv environment is installed
if [ ! -d "dataset/.venv" ]; then
    echo "uv environment not found in dataset. Installing uv environment..."
    cd dataset
    uv venv --python 3.12
    uv pip install -r requirements.txt
    cd ..
else
    echo "uv environment already exists"
fi

# install uv enivironment in evals/PPE if not already installed, otherwise just say uv environment is installed
if [ ! -d "evals/PPE/.venv" ]; then
    echo "uv environment not found in evals/PPE. Installing uv environment..."
    cd evals/PPE
    uv venv --python 3.12
    uv pip install -r requirements.txt
    uv pip install --no-build-isolation flash-attn==2.7.1.post4
    cd ../../
else
    echo "uv environment already exists in evals/PPE"
fi

# install uv environment in evals/rewardbench if not already installed, otherwise just say uv environment is installed
if [ ! -d "evals/reward-bench/.venv" ]; then
    echo "uv environment not found in evals/reward-bench. Installing uv environment..."
    cd evals/reward-bench
    uv venv --python 3.12
    uv pip install -e .[generative]
    uv pip install --no-build-isolation flash-attn==2.7.1.post4
    uv pip install huggingface_hub[hf_transfer]
    cd ../../
else
    echo "uv environment already exists in evals/reward-bench"
fi

# install uv environment in train/OpenRLHF if not already installed, otherwise just say uv environment is installed
if [ ! -d "train/OpenRLHF/.venv" ]; then
    echo "uv environment not found in train/OpenRLHF. Installing uv environment..."
    cd train/OpenRLHF
    uv venv --python 3.12
    uv pip install -e .[vllm]
    uv pip install torch setuptools wheel psutil
    uv pip install --no-build-isolation flash-attn==2.7.1.post4
    uv pip install huggingface_hub[hf_transfer] liger-kernel
    cd ../../
else
    echo "uv environment already exists in train/OpenRLHF"
fi
