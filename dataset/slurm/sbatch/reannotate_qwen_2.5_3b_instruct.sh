#!/bin/bash
#SBATCH --job-name=reannotate_qwen_2.5_3b_instruct
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

python reannotate_ratings.py --model_name Qwen/Qwen2.5-3B-Instruct \
                             --num_gpus 4 \
                             --port 8000 \
                             --dataset_name nectar \
                             --k_val 7 \
                             --slurm_task_id $SLURM_ARRAY_TASK_ID \
                             --slurm_num_tasks $SLURM_ARRAY_TASK_COUNT \
                             --num-processes 32 \
                             --seed 42 
