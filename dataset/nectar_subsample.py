from datasets import load_dataset

weak_preferences_dataset = load_dataset("esfrankel17/Nectar_binarized_w_weak_preferences", split="rank")

# randomly sample 10% of the dataset
sampled_dataset = weak_preferences_dataset.shuffle(seed=1000).select(range(int(len(weak_preferences_dataset) * 0.1)))

sampled_dataset.push_to_hub("esfrankel17/Nectar_10_pct_subsample_binarized_w_weak_preferences_cleaned", split="rank")