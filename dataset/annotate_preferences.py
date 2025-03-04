from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset


def main(args):
    log_dir = Path(args.log_dir) / "dataset" / "annotate_preferences" / args.dataset_name / args.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(args, file=open(log_dir / "args.txt", "w"))
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True, default="logs")
    parser.add_argument("--seed", type=int, required=True, default=42)
    args = parser.parse_args()
    main(args)

