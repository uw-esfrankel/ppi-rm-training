import os
import pickle
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np


MODEL_SIZES = {
    "Qwen": [0.5, 1.5, 3, 7, 14, 32, 72],
    "Llama": [8, 70],
    "Gemma": [27]
}

BASE_DATASETS = ["HelpSteer", "HelpSteer2", "Nectar", "ChatbotArena55k", "UltraFeedback"]


def get_hf_dataset_name(base_dataset_name: str, model_name: str, model_size: float) -> str:
    if model_name == "Qwen":
        return f"esfrankel17/original_{base_dataset_name}_binarized_Qwen2.5-{model_size}B-Instruct_preferences"
    elif model_name == "Llama":
        return f"esfrankel17/original_{base_dataset_name}_binarized_Llama-3.1-{model_size}B-Instruct_preferences"
    elif model_name == "Gemma":
        return f"esfrankel17/original_{base_dataset_name}_binarized_gemma-2-{model_size}b-it_preferences"
    

def get_formatted_model_name(model_name: str, model_size: float) -> str:
    if model_name == "Qwen":
        return f"Qwen2.5-{model_size}B-Instruct"
    elif model_name == "Llama":
        return f"Llama-3.1-{model_size}B-Instruct"
    elif model_name == "Gemma":
        return f"Gemma-2-{model_size}B-Instruct"
    

def main():
    missing_datasets = []

    if os.path.exists("dataset_agreement_rates.pkl"):
        with open("dataset_agreement_rates.pkl", "rb") as f:
            dataset_agreement_rates = pickle.load(f)
    else:
        dataset_agreement_rates = dict()

    for base_dataset_name in BASE_DATASETS:
        if base_dataset_name not in dataset_agreement_rates:
            dataset_agreement_rates[base_dataset_name] = dict()
            
        for model_name, model_sizes in MODEL_SIZES.items():
            for model_size in model_sizes:
                hf_dataset_name = get_hf_dataset_name(base_dataset_name, model_name, model_size)
                formatted_model_name = get_formatted_model_name(model_name, model_size)

                # Check if we already have results for this model in any split
                splits_complete = True
                try:
                    dataset = load_dataset(hf_dataset_name)
                    for split in dataset.keys():
                        if (split not in dataset_agreement_rates[base_dataset_name] or 
                            formatted_model_name not in dataset_agreement_rates[base_dataset_name][split]):
                            splits_complete = False
                            break
                except Exception as e:
                    print(f"Error loading dataset {hf_dataset_name}: {e}")
                    missing_datasets.append(hf_dataset_name)
                    continue

                if splits_complete:
                    print(f"Skipping {formatted_model_name} for {base_dataset_name} - already processed")
                    continue

                # Process the dataset if not already complete
                for split in dataset.keys():
                    if (split not in dataset_agreement_rates[base_dataset_name] or 
                        formatted_model_name not in dataset_agreement_rates[base_dataset_name][split]):
                        data_split = dataset[split]
                        agreement_rate = sum(data_split["model_agreed_with_original"]) / len(data_split["model_agreed_with_original"])
                        print(f"Agreement rate for {formatted_model_name} on split {split} for dataset {base_dataset_name}: {agreement_rate}")

                        # select rows in data_split where model_agreed_with_original is False
                        data_split_disagreement = data_split.filter(lambda x: not x["model_agreed_with_original"])
                        # get all of the values of the difference between "original_chosen_rating" and "original_rejected_rating" on the disagreement rows
                        disagreement_ratings = [data_split_disagreement[i]["original_chosen_rating"] - data_split_disagreement[i]["original_rejected_rating"] for i in range(len(data_split_disagreement))]
                        # the disagreement ratings are negative if the base_dataset_name is Nectar
                        if base_dataset_name == "Nectar":
                            disagreement_ratings = [-x for x in disagreement_ratings]
                        avg_disagreement_rating = np.mean(disagreement_ratings)
                        print(f"Average disagreement rating for {formatted_model_name} on split {split} for dataset {base_dataset_name}: {avg_disagreement_rating}")
                        
                        if split not in dataset_agreement_rates[base_dataset_name]:
                            dataset_agreement_rates[base_dataset_name][split] = {}
                        dataset_agreement_rates[base_dataset_name][split][formatted_model_name] = (agreement_rate, model_size, model_name, avg_disagreement_rating, disagreement_ratings)

    with open("dataset_agreement_rates.pkl", "wb") as f:
        pickle.dump(dataset_agreement_rates, f)

    for base_dataset_name in BASE_DATASETS:
        for split in dataset_agreement_rates[base_dataset_name].keys():
            plt.figure(figsize=(10, 6))
            
            # Group by model name for plotting
            model_data = {}
            for _, (agreement_rate, model_size, model_name, avg_disagreement_rating, disagreement_ratings) in dataset_agreement_rates[base_dataset_name][split].items():
                if model_name not in model_data:
                    model_data[model_name] = {"sizes": [], "rates": [], "avg_disagreement_ratings": [], "disagreement_ratings": []}
                model_data[model_name]["sizes"].append(model_size)
                model_data[model_name]["rates"].append(agreement_rate)
                model_data[model_name]["avg_disagreement_ratings"].append(avg_disagreement_rating)
                model_data[model_name]["disagreement_ratings"].append(disagreement_ratings)
            
            # Plot each model series with different colors
            colors = {'Qwen': 'blue', 'Llama': 'red', 'Gemma': 'green'}
            
            for model_name, data in model_data.items():
                # Sort by model size to ensure correct line plotting
                sorted_indices = np.argsort(data["sizes"])
                sizes = [data["sizes"][i] for i in sorted_indices]
                rates = [data["rates"][i] for i in sorted_indices]
                
                plt.plot(sizes, rates, 'o-', label=model_name, color=colors.get(model_name, 'black'))
            
            plt.xscale('log')  # Set x-axis to log scale
            plt.xlabel('Model Size (B)')
            plt.ylabel('Agreement Rate')
            plt.title(f'Agreement Rate vs Model Size for {base_dataset_name} ({split})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the figure
            os.makedirs(os.path.join("figures", base_dataset_name), exist_ok=True)
            plt.savefig(os.path.join("figures", base_dataset_name, f"{split}_agreement_rates.png"), dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            for model_name, data in model_data.items():
                plt.plot(data["sizes"], data["avg_disagreement_ratings"], 'o-', label=model_name, color=colors.get(model_name, 'black'))
            plt.xscale('log')
            plt.xlabel('Model Size (B)')
            plt.ylabel('Average Disagreement Rating')
            plt.title(f'Average Disagreement Rating vs Model Size for {base_dataset_name} ({split})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join("figures", base_dataset_name, f"{split}_avg_disagreement_ratings.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # plot histogram of the differences
            plt.figure(figsize=(10, 6))
            for model_name, data in model_data.items():
                # Flatten all disagreement ratings for this model
                all_disagreements = [rating for ratings in data["disagreement_ratings"] for rating in ratings]
                plt.hist(all_disagreements, label=model_name, color=colors.get(model_name, 'black'), 
                        alpha=0.5, density=True)  # Added alpha for transparency and density=True for normalization
            plt.xlabel('Difference in Ratings')
            plt.ylabel('Density')  # Changed to Density since we normalized the histograms
            plt.title(f'Distribution of Rating Differences for {base_dataset_name} ({split})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join("figures", base_dataset_name, f"{split}_disagreement_ratings_histogram.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Missing datasets: {missing_datasets}")
if __name__ == "__main__":
    main()