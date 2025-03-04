import argparse
from pathlib import Path

from datasets import load_dataset

from reannotation.nectar import reannotate_nectar
from utils.vllm_manager import VLLMManager


def main(args):
    slurm_task_id = args.slurm_task_id
    slurm_num_tasks = args.slurm_num_tasks
    num_processes = args.num_processes
    num_data_points = args.num_data_points
    seed = args.seed
    
    dirname = Path(args.dirname) / "reannotate_dataset" / args.dataset_name / args.model_name.split("/")[-1] / f"slurm-{slurm_task_id}"
    dirname.mkdir(parents=True, exist_ok=True)
    
    print(args, file=open(dirname / "args.txt", "w"))
    
    vllm_manager = VLLMManager(args.model_name, num_gpus=args.num_gpus, port=args.port, seed=args.seed, vllm_process_outdir=dirname)
    
    assert slurm_task_id < slurm_num_tasks, f"slurm_task_id {slurm_task_id} is greater than or equal to slurm_num_tasks {slurm_num_tasks}"
    
    if args.dataset_name == "nectar":
        reannotate_nectar(
            dirname=dirname,
            vllm_manager=vllm_manager,
            slurm_task_id=slurm_task_id,
            slurm_num_tasks=slurm_num_tasks,
            num_processes=num_processes,
            num_data_points=num_data_points,
            seed=seed,
            k_val=args.k_val,
            explain=args.explain,
            do_pairwise=args.do_pairwise,
            shuffle=args.shuffle
        )
    elif args.dataset_name == "uf":
        reannotate_uf(args)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # vllm arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--write_to_logfile", default=False, action=argparse.BooleanOptionalAction)

    # dataset arguments
    parser.add_argument("--dataset_name", choices=["nectar", "uf"], type=str, required=True)
    parser.add_argument("--dirname", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    
    # nectar-specific arguments
    parser.add_argument('--num-processes', type=int, default=64, help='Number of parallel process to spawn')
    parser.add_argument('--num_data_points', '-n', type=int, help='Number of datapoints to annotate per process')
    parser.add_argument('--k_val', '-k', type=int, help='K for K-wise: gives the number of responses per prompt')
    parser.add_argument('--explain', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--do-pairwise', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--shuffle', default=False, action=argparse.BooleanOptionalAction)
    
    # slurm arguments
    parser.add_argument('--slurm_task_id', type=int, required=True)
    parser.add_argument('--slurm_num_tasks', type=int, required=True)
    
    args = parser.parse_args()
    main(args)