import pandas as pd
from datasets import load_dataset, Dataset
from scipy import datasets


MODEL_SIZES = {
    "Qwen": [0.5, 1.5, 3, 7, 14, 32, 72],
    "Llama": [8, 70],
    "Gemma": [27]
}

BASE_DATASETS = ["HelpSteer", "HelpSteer2", "Nectar", "ChatbotArena55k", "UltraFeedback"]


def get_full_model_name(model_name: str, model_size: float) -> str:
    if model_name == "Qwen":
        return f"Qwen2.5-{model_size}B-Instruct"
    elif model_name == "Llama":
        return f"Llama-3.1-{model_size}B-Instruct"
    elif model_name == "Gemma":
        return f"gemma-2-{model_size}b-it"


def get_hf_dataset_name(base_dataset_name: str, model_name: str, model_size: float) -> str:
    if model_name == "Qwen":
        return f"esfrankel17/original_{base_dataset_name}_binarized_Qwen2.5-{model_size}B-Instruct_preferences"
    elif model_name == "Llama":
        return f"esfrankel17/original_{base_dataset_name}_binarized_Llama-3.1-{model_size}B-Instruct_preferences"
    elif model_name == "Gemma":
        return f"esfrankel17/original_{base_dataset_name}_binarized_gemma-2-{model_size}b-it_preferences"


def main():
    for base_dataset_name in BASE_DATASETS:
        original_binarized_dataset = load_dataset(f"esfrankel17/original_{base_dataset_name}_binarized")
        original_binarized_dataset = original_binarized_dataset.remove_columns(["chosen_rating", "rejected_rating"])

        for split in original_binarized_dataset.keys():
            merged_split = original_binarized_dataset[split].to_pandas()
            original_num_rows = merged_split.shape[0]

            merged_split["chosen_str"] = merged_split["chosen"].apply(str)
            merged_split["rejected_str"] = merged_split["rejected"].apply(str)

            for model_name, model_sizes in MODEL_SIZES.items():
                for model_size in model_sizes:
                    annotated_dataset_name = get_hf_dataset_name(base_dataset_name, model_name, model_size)
                    full_model_name = get_full_model_name(model_name, model_size).lower().replace("-", "_")
                    annotated_split = load_dataset(annotated_dataset_name, split=split).to_pandas()

                    annotated_split["chosen_str"] = annotated_split["original_chosen"].apply(str)
                    annotated_split["rejected_str_original"] = annotated_split["original_rejected"].apply(str)
                    annotated_split["model_chosen_str"] = annotated_split["model_chosen"].apply(str)

                    # merge on the chosen_str
                    combined_split = pd.merge(
                        merged_split,
                        annotated_split,
                        on="chosen_str",
                        how="left"
                    )

                    num_rows_with_nans = combined_split["rejected_str_original"].isna().sum()

                    # assert that the number of rows with nans in combined_split is the difference between the number of rows in merged_split and annotated_split
                    assert (num_rows_with_nans == 
                            (original_num_rows - annotated_split.shape[0])), f"The number of rows with nans in combined_split is not the difference between the number of rows in merged_split and annotated_split for {annotated_dataset_name} and {split}"
                    
                    # assert that rejected strings match for non-null values
                    non_null_mask = ~pd.isna(combined_split["rejected_str_original"])
                    assert (combined_split.loc[non_null_mask, "rejected_str_original"] == 
                            combined_split.loc[non_null_mask, "rejected_str"]).all()
                    
                    # create mask for rows where model_agreed_with_original is not na and is false
                    model_agreed_with_original_mask = (combined_split["model_agreed_with_original"].notna() & (combined_split["model_agreed_with_original"] == False))
                    assert (combined_split.loc[model_agreed_with_original_mask, "model_chosen_str"] == 
                            combined_split.loc[model_agreed_with_original_mask, "rejected_str_original"]).all()
                    model_disagreed_with_original_mask = (combined_split["model_agreed_with_original"].notna() & (combined_split["model_agreed_with_original"] == True))
                    assert (combined_split.loc[model_disagreed_with_original_mask, "model_chosen_str"] == 
                            combined_split.loc[model_disagreed_with_original_mask, "chosen_str"]).all()

                    combined_split = combined_split[['chosen_str', 'model_agreed_with_original']]
                    # convert the model_agreed_with_original column to an int for all rows where it is not na
                    combined_split.loc[combined_split["model_agreed_with_original"].notna(), "model_agreed_with_original"] = combined_split.loc[combined_split["model_agreed_with_original"].notna(), "model_agreed_with_original"].astype(int)

                    # more sanity checks
                    # assert that model_agreed_with_original is 0 or 1 for all rows where it is not na
                    assert combined_split.loc[combined_split["model_agreed_with_original"].notna(), "model_agreed_with_original"].isin([0, 1]).all()
                    # assert that the number of rows with nans in model_agreed_with_original is the same as the number of rows with nans in rejected_str_original
                    assert combined_split["model_agreed_with_original"].isna().sum() == num_rows_with_nans
                    # should have same number of rows as original_binarized_dataset
                    assert combined_split.shape[0] == original_num_rows

                    # rename the model_agreed_with_original column to {full_model_name}_agreement
                    combined_split.rename(columns={"model_agreed_with_original": f"{full_model_name}_agreement"}, inplace=True)

                    merged_split = pd.merge(
                        merged_split,
                        combined_split,
                        on="chosen_str",
                        how="left"
                    )

            assert merged_split.shape[0] == original_num_rows
            merged_split = merged_split.drop(columns=["chosen_str", "rejected_str"])
            assert len(merged_split.columns) == 3 + sum(len(MODEL_SIZES[model_name]) for model_name in MODEL_SIZES.keys())
            
            # convert to dataset
            merged_split = Dataset.from_pandas(merged_split)
            merged_split.push_to_hub(f"esfrankel17/{base_dataset_name}_binarized_w_weak_preferences", split=split)
                    

if __name__ == "__main__":
    main()