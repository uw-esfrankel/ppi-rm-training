import argparse
import random
from datasets import load_dataset, Dataset
from huggingface_hub import delete_repo, repo_exists
import numpy as np

from utils.message_utils import parse_out_prompt_turns_hh_format
from utils.torch_utils import set_seed_everywhere


VANILLA_DATASETS = [
    "nvidia/HelpSteer",
    "nvidia/HelpSteer2",
    "openbmb/UltraFeedback",
    "berkeley-nest/Nectar"
]

DATASET_TO_KEYS = {
    "nvidia/HelpSteer": ["average_rating", "average_rating_no_verbosity", "average_rating_no_verbosity_no_complexity", "goodness_score"],
    "nvidia/HelpSteer2": ["average_rating", "average_rating_no_verbosity", "average_rating_no_verbosity_no_complexity", "goodness_score"],
    "berkeley-nest/Nectar": ["rank"],
    "openbmb/UltraFeedback": ["average_rating", "overall_score"],
}

# **************************************************************

def compute_average_rating(row):
    ratings = [row[k] for k in ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]]
    row["average_rating"] = np.mean(ratings)
    return row

def compute_average_rating_no_verbosity(row):
    ratings = [row[k] for k in ["helpfulness", "correctness", "coherence", "complexity"]]
    row["average_rating_no_verbosity"] = np.mean(ratings)
    return row

def compute_average_rating_no_verbosity_no_complexity(row):
    ratings = [row[k] for k in ["helpfulness", "correctness", "coherence"]]
    row["average_rating_no_verbosity_no_complexity"] = np.mean(ratings)
    return row

# following the goodness score formula from the paper
def compute_rating_goodness_score(row):
    ratings = np.array([row[k] for k in ["helpfulness", "correctness", "coherence"]])
    row["goodness_score"] = np.array([0.65, 0.8, 0.45]).dot(ratings)
    return row

def compute_all_ratings_helpsteer(row):
    row = compute_average_rating(row)
    row = compute_average_rating_no_verbosity(row)
    row = compute_average_rating_no_verbosity_no_complexity(row)
    row = compute_rating_goodness_score(row)
    return row

def compute_average_rating_ultrafeedback(row):
    for completion in row["completions"]:
        ratings = []
        for aspect in ["instruction_following", "honesty", "truthfulness", "helpfulness"]:
            ratings.append(int(completion["annotations"][aspect]["Rating"]))
        completion.update({"average_rating": sum(ratings) / 4})
    return row
    
# **************************************************************


DATASET_TO_COMPUTE_FUNCTIONS = {
    "nvidia/HelpSteer": compute_all_ratings_helpsteer,
    "nvidia/HelpSteer2": compute_all_ratings_helpsteer,
    "berkeley-nest/Nectar": lambda row: row,
    "openbmb/UltraFeedback": compute_average_rating_ultrafeedback,
}

def group_and_sort(dataset, dataset_key, dataset_name):
    grouped_data = {}
    for row in dataset:
        if dataset_name == "openbmb/UltraFeedback":
            prompt = row["instruction"]
        else:
            prompt = row["prompt"]
        
        if dataset_name == "openbmb/UltraFeedback":
            grouped_data[prompt] = row["completions"]
        elif dataset_name == "berkeley-nest/Nectar":
            grouped_data[prompt] = row["answers"]
        else:   
            if prompt not in grouped_data:
                grouped_data[prompt] = []
            grouped_data[prompt].append(row)
    for prompt in grouped_data:
        grouped_data[prompt].sort(key=lambda x: x[dataset_key], reverse=dataset_name != "berkeley-nest/Nectar")
    return grouped_data 

def load_and_process_dataset(dataset_name):
    if dataset_name == "nvidia/HelpSteer" or dataset_name == "nvidia/HelpSteer2" or dataset_name == "berkeley-nest/Nectar":
        dataset = load_dataset(dataset_name)['train']
        return dataset
    else:
        assert dataset_name == "openbmb/UltraFeedback"
        dataset_a = load_dataset(dataset_name, split="train")
        dataset_b = load_dataset("truthful_qa", "generation", split="validation")
        dataset_c = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        dataset_a = dataset_a.remove_columns(["models", "correct_answers", "incorrect_answers"])
        dataset_a = dataset_a.filter(lambda x: x["source"] != "truthful_qa")
        print(f"Remaining samples after removing the TruthfulQA source [{dataset_a.num_rows} / 63967]")

        contaminated_prompts = list(set(dataset_b["question"] + dataset_c["question"]))
        dataset_a = dataset_a.filter(lambda x: x["instruction"] not in contaminated_prompts)
        print(f"Remaining samples after removing the contaminated prompts [{dataset_a.num_rows} / 63967]")

        del dataset_b, dataset_c
        
        def remove_partially_annotated(dataset_row: dict) -> dict:
            completions = []
            for completion in dataset_row["completions"]:
                if not all(aspect in ["instruction_following", "honesty", "truthfulness", "helpfulness"] for aspect in completion["annotations"].keys()):
                    continue
                ratings = []
                for aspect in ["instruction_following", "honesty", "truthfulness", "helpfulness"]:
                    try:
                        ratings.append(int(completion["annotations"][aspect]["Rating"]))
                    except:
                        break
                if len(ratings) != 4:
                    continue
                if "critique" in completion:
                    completion.pop("critique")
                if "custom_system_prompt" in completion:
                    completion.pop("custom_system_prompt")
                if "principle" in completion:
                    completion.pop("principle")
                completions.append(completion)
            dataset_row["completions"] = completions
            return dataset_row
        
        dataset_a = dataset_a.map(remove_partially_annotated)
        print(f"Remaining samples after removing ones partially annotated or with invalid values [{dataset_a.num_rows} / 63967]")
        
        dataset_a = dataset_a.filter(lambda x: len(x["completions"]) > 1)
        print(f"Remaining samples after removing the rows with less than 2 annotations (at least one to be chosen and one to be rejected) [{dataset_a.num_rows} / 63967]")
        
        return dataset_a

def binarize_dataset(dataset, dataset_name, dataset_keys, compute_function):
    dataset = dataset.map(compute_function)
    data_name = dataset_name.split("/")[-1]

    for dataset_key in dataset_keys:
        grouped_data = group_and_sort(dataset, dataset_key, dataset_name)
        
        chosen_list = []
        rejected_list = []
        chosen_rating_list = []
        rejected_rating_list = []
        prompts_list = []
        
        for prompt, samples in grouped_data.items():
            # Filtering out prompts with only one completion or with only the same rating
            if len(samples) < 2 or len(set(row[dataset_key] for row in samples)) == 1:
                continue
            
            chosen = samples[0]
            chosen_score = chosen[dataset_key]
            
            # Choose rejected randomly from the remaining samples
            # Ensure rejected doesn't have the same score in dataset_key as chosen
            remaining_samples = [s for s in samples[1:] if s[dataset_key] != chosen_score]
            if not remaining_samples:
                continue  # Skip if no suitable rejected sample found
            rejected = random.choice(remaining_samples)
            rejected_score = rejected[dataset_key]
            
            if dataset_name == "berkeley-nest/Nectar":
                initial_turns = parse_out_prompt_turns_hh_format(prompt)
                chosen_msgs = initial_turns + [
                    {'role': 'assistant', 'content': chosen['answer']},
                ]
                rejected_msgs = initial_turns + [
                    {'role': 'assistant', 'content': rejected['answer']},
                ]
            else:
                chosen_msgs =  [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': chosen['response']},
                ]
                rejected_msgs =  [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': rejected['response']},
                ]
            
            chosen_list.append(chosen_msgs)
            rejected_list.append(rejected_msgs)
            chosen_rating_list.append(chosen_score)
            rejected_rating_list.append(rejected_score)
            prompts_list.append(prompt)
            
        binarized_dataset_split = Dataset.from_dict({
            "prompt": prompts_list,
            "chosen": chosen_list,
            "chosen_rating": chosen_rating_list,
            "rejected": rejected_list,
            "rejected_rating": rejected_rating_list
        })
        
        binarized_dataset_split.push_to_hub(repo_id=f"esfrankel17/original_{data_name}_binarized", split=f"{dataset_key}")
        print(f"Pushed {data_name} with split {dataset_key} to the hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate_all", action="store_true", help="Delete all existing datasets and re-process them")
    args = parser.parse_args()
        
    set_seed_everywhere(args.seed)
    for dataset_name in VANILLA_DATASETS:
        data_name = dataset_name.split("/")[-1]
        repo_id = f"esfrankel17/original_{data_name}_binarized"
        if repo_exists(repo_id=repo_id, repo_type="dataset"):
            if args.recreate_all:
                print(f"Deleting {repo_id} because --recreate_all was passed")
                delete_repo(repo_id=repo_id, repo_type="dataset")
            else:
                print(f"Skipping {dataset_name} because it has already been processed")
                continue
        else:
            print(f"Processing {dataset_name}, since {repo_id} does not exist")
        
        dataset = load_and_process_dataset(dataset_name)
        dataset_keys = DATASET_TO_KEYS[dataset_name]
        compute_function = DATASET_TO_COMPUTE_FUNCTIONS[dataset_name]
        binarize_dataset(dataset, dataset_name, dataset_keys, compute_function)
