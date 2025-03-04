sbatch --output=slurm/logs/reannotate_qwen_2.5_14b_instruct/%A_%a.out \
       --error=slurm/logs/reannotate_qwen_2.5_14b_instruct/%A_%a.err \
       slurm/sbatch/reannotate_qwen_2.5_14b_instruct.sh