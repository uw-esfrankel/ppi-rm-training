# Dataset Annotation

## Overview

This folder contains code for attaining the pseudolabels for the following datasets:
- [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- [Chatbot Arena Human Preferences](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k)
- [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)

This folder performs the following actions:
- Provides wholesale reannotation of a dataset with a given LLM (for Nectar and UltraFeedback only)
- Dataset binarization (for HelpSteer, HelpSteer2, Nectar, and UltraFeedback)
- Provides preference elicitation for pairwise comparisons (all datasets)

## Setup
Assuming the setup script has been run in the upper directory, we can run:
```bash
source .venv/bin/activate
```

If in Hyak, we also run:
```bash
module load cuda/12.4.1 
module load gcc/9.3.0
```

## Reannotation

To reannotate a dataset, we can run:
```bash
python reannotate_ratings.py --model_name <model_name> --dataset_name <dataset_name> --output_dir <output_dir>
```

