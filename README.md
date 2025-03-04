# Semi-Supervised LLM Preference Training

## Get Started
First, install uv as a package manager:
```
# assuming linux environment
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Hyak
If on Hyak and on a compute node, run the following:
```
module load cuda/12.4.1 
module load gcc/9.3.0
```

### Running setup script
To install all dependencies, run the following:
```
bash scripts/setup_env_hyak.sh
```
