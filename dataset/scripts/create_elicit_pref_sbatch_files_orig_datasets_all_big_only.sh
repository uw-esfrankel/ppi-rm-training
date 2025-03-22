#! /bin/bash

qwen_model_sizes=()
llama_model_sizes=("70")
gemma_model_sizes=()

get_num_gpus() {
    local size=$1
    local num_gpus
    if (( $(echo "$size > 32" | bc -l) )); then
        num_gpus=8
    elif (( $(echo "$size >= 14" | bc -l) )); then
        num_gpus=2
    else
        num_gpus=1
    fi
    echo $num_gpus
}

model_names=()
num_gpus_reqs=()
# Add all Qwen models to the model_name_size_reqs array
for size in "${qwen_model_sizes[@]}"; do
    num_gpus=$(get_num_gpus "$size")
    model_names+=("Qwen/Qwen2.5-${size}B-Instruct")
    num_gpus_reqs+=($num_gpus)
done

for size in "${llama_model_sizes[@]}"; do
    num_gpus=$(get_num_gpus "$size")
    model_names+=("meta-llama/Llama-3.1-${size}B-Instruct")
    num_gpus_reqs+=($num_gpus)
done

for size in "${gemma_model_sizes[@]}"; do
    num_gpus=$(get_num_gpus "$size")
    model_names+=("google/gemma-2-${size}b-it")
    num_gpus_reqs+=($num_gpus)
done

# Declare two separate arrays that we'll iterate over in parallel
dataset_names=("Nectar")
num_slurm_jobs=(100)

# Now setting up folders as needed
mkdir -p slurm/elicit_pref_orig_datasets
mkdir -p slurm/elicit_pref_orig_datasets/logs
mkdir -p slurm/elicit_pref_orig_datasets/sbatch

# Create the sbatch files looping over the datasets and models
for ((j=0; j<${#model_names[@]}; j++)); do
for ((i=0; i<${#dataset_names[@]}; i++)); do

model_name=${model_names[$j]}
num_gpus=${num_gpus_reqs[$j]}
dataset_name=${dataset_names[$i]}
num_slurm_jobs=${num_slurm_jobs[$i]}

# Model identifier
model_identifier=${model_name//\//_}

hf_dataset_name="esfrankel17/original_${dataset_name}_binarized"

# Create the folder if it doesn't exist
mkdir -p slurm/elicit_pref_orig_datasets/logs/${dataset_name}/${model_identifier}
mkdir -p slurm/elicit_pref_orig_datasets/sbatch/${dataset_name}

file_content="#!/bin/bash
#SBATCH --job-name=elicit_pref_${dataset_name}_${model_identifier}
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=ericsf@cs.washington.edu
#SBATCH --account=sewoong
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=${num_gpus}
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --time=1-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/gscratch/sewoong/ericsf/ppi-rm-training/dataset
#SBATCH --export=all
#SBATCH --output=/gscratch/sewoong/ericsf/ppi-rm-training/dataset/slurm/elicit_pref_orig_datasets/logs/${dataset_name}/${model_identifier}/%x_%A_%a.out
#SBATCH --error=/gscratch/sewoong/ericsf/ppi-rm-training/dataset/slurm/elicit_pref_orig_datasets/logs/${dataset_name}/${model_identifier}/%x_%A_%a.err
#SBATCH --array=0-$((num_slurm_jobs-1))

module load cuda/12.4.1 
module load gcc/9.3.0

source .venv/bin/activate

python elicit_preferences.py --model_name ${model_name} --num_gpus ${num_gpus} --dataset_name ${hf_dataset_name}
"

printf '%s\n' "$file_content" > slurm/elicit_pref_orig_datasets/sbatch/${dataset_name}/elicit_pref_${model_identifier}.sbatch
sbatch slurm/elicit_pref_orig_datasets/sbatch/${dataset_name}/elicit_pref_${model_identifier}.sbatch

done
done