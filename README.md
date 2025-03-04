# Semi-Supervised LLM Preference Training

## Get Started
First, install uv as a package manager:
```
# assuming linux environment
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Hyak
If on Hyak, just run the setup script:
```
bash scripts/setup_env_hyak.sh
```

## Eliciting Preferences
First, make sure the environment is set up:
```
cd dataset
uv venv --python 3.10
uv pip install -r requirements.txt
```
### Re-annotating Datasets



### Environment Setup
```
module load cuda/12.4.1 
module load gcc/9.3.0
conda env create -f env.yml -y
conda activate ppi-rm-training
python -m pip install -r PPE/requirements.txt
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.1.post4 --no-build-isolation
cd OpenRLHF
python -m pip install -e .[vllm]
```

## Baselines
```
module load cuda/12.4.1 
module load gcc/9.3.0
conda activate ppi-rm-training
sh OpenRLHF/examples/scripts/train_rm_llama.sh
```