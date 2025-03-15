#!/bin/bash
if [ ! -d "dataset/.venv" ]; then
    echo "uv environment not found in dataset. Installing uv environment..."
    cd dataset
    uv venv --python 3.12
    uv pip install -r requirements.txt
    cd ..
else
    echo "uv environment already exists"
fi