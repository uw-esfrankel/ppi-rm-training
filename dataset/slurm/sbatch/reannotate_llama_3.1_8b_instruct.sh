#!/bin/bash
#SBATCH --job-name=reannotate_all
#SBATCH --mail-type=ARRAY_TASKS,FAIL,INVALID_DEPEND
#SBATCH --mail-user=ericsf@cs.washington.edu
#SBATCH --account=sewoong
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gpus=4
#SBATCH --constraint=a100|l40s|l40|a40
#SBATCH --time=08:00:00
#SBATCH --chdir=/gscratch/sewoong/ericsf/ppi-rm-training/dataset
#SBATCH --export=all
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --array=0-19

module load cuda/12.4.1 
module load gcc/9.3.0

source .venv/bin/activate

python reannotate_ratings.py --model_name meta-llama/Llama-3.1-8B-Instruct --num_gpus 4 --dirname reannotate_dataset/llama_3.1_8b_instruct --seed 42 --k_val 10 --shuffle True --explain True --do_pairwise True --slurm_job_id $SLURM_ARRAY_JOB_ID --num_jobs $SLURM_ARRAY_TASK_COUNT