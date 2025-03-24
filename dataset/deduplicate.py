# Redux of https://gist.github.com/natolambert/1aed306000c13e0e8c5bc17c1a5dd300
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
import numpy as np
import json
from datetime import datetime
import glob


BASE_DATASETS = ["HelpSteer", "HelpSteer2", "Nectar", "ChatbotArena55k", "UltraFeedback"]

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_matches_file = f'best_matches_{timestamp}.json'
    json_contaminated_file = f'contaminated_indices_{timestamp}.json'

    # Check for existing JSON files
    existing_matches_files = glob.glob('best_matches_*.json')
    existing_contaminated_files = glob.glob('contaminated_indices_*.json')

    if existing_matches_files and existing_contaminated_files:
        print("Found existing analysis files. Loading them instead of recomputing...")
        # Load the most recent files (they should be paired by timestamp)
        latest_matches_file = max(existing_matches_files)
        latest_contaminated_file = max(existing_contaminated_files)
        
        with open(latest_matches_file, 'r', encoding='utf-8') as f:
            best_matches = json.load(f)
        with open(latest_contaminated_file, 'r', encoding='utf-8') as f:
            contaminated_indices = json.load(f)
    else:
        contaminated_indices = {dataset: {} for dataset in BASE_DATASETS}
        best_matches = []
        
        rb_dataset = load_dataset("allenai/reward-bench", split="filtered")
        reward_bench_prompts = rb_dataset["prompt"]
        
        for base_dataset_name in BASE_DATASETS:
            print(f"\nAnalyzing dataset: {base_dataset_name}")
            pref_dataset = load_dataset(f"esfrankel17/{base_dataset_name}_binarized_w_weak_preferences")
            
            # Initialize with lists instead of sets
            contaminated_indices[base_dataset_name] = {split: [] for split in pref_dataset.keys()}
            
            for split in pref_dataset.keys():
                print(f"\nProcessing split: {split}")
                pref_dataset_split = pref_dataset[split]
                pref_dataset_prompts = pref_dataset_split["prompt"]
                
                # Vectorize all prompts
                print("Vectorizing prompts...")
                vectorizer = CountVectorizer(ngram_range=(7,13))
                all_prompts = list(reward_bench_prompts) + list(pref_dataset_prompts)
                vectorized = vectorizer.fit_transform(tqdm(all_prompts))
                
                # Split vectorized matrix back into two datasets
                n_rb = len(reward_bench_prompts)
                rb_vectorized = vectorized[:n_rb]
                pref_vectorized = vectorized[n_rb:]
                
                # Calculate similarity matrix
                print("Calculating similarities...")
                similarity_matrix = (rb_vectorized @ pref_vectorized.T).toarray()
                
                # Update contaminated indices tracking (use list instead of set)
                for rb_idx in tqdm(range(similarity_matrix.shape[0]), desc="Finding matches"):
                    matches = np.where(similarity_matrix[rb_idx] > 0)[0].tolist()  # Convert to list
                    contaminated_indices[base_dataset_name][split].extend(matches)  # Use extend instead of update
                
                # Make sure indices are unique before saving
                contaminated_indices[base_dataset_name][split] = list(set(contaminated_indices[base_dataset_name][split]))
                
                # Find best matching pairs for each RewardBench prompt
                for rb_idx in tqdm(range(similarity_matrix.shape[0]), desc="Finding best matches"):
                    # Find the best matching pref prompt for this RewardBench prompt
                    best_pref_idx = np.argmax(similarity_matrix[rb_idx])
                    best_score = similarity_matrix[rb_idx, best_pref_idx]
                    
                    # Only include if there is actually an overlap (score > 0)
                    if best_score > 0:
                        rb_prompt = reward_bench_prompts[rb_idx]
                        pref_prompt = pref_dataset_prompts[best_pref_idx]

                        # full pref
                        pref_dict = pref_dataset_split.select([best_pref_idx]).to_dict()
                        
                        best_matches.append({
                            'reward_bench_id': rb_dataset.select([rb_idx])['id'][0],
                            'pref_idx': int(best_pref_idx),  # Convert numpy int to regular int for JSON
                            'reward_bench_prompt': rb_prompt,
                            'pref_prompt': pref_prompt,
                            'overlap_score': float(best_score),  # Convert to float for JSON serialization
                            # 'reward_bench_full': dict(rb_full),
                            'pref_full': pref_dict,
                            'pref_subset': base_dataset_name
                        })
                        
                # Clear memory after processing each split
                del vectorizer, vectorized, rb_vectorized, pref_vectorized, similarity_matrix
                
        # Save matches results
        with open(json_matches_file, 'w', encoding='utf-8') as f:
            json.dump(best_matches, f, ensure_ascii=False, indent=2)
        # Save contaminated indices
        with open(json_contaminated_file, 'w', encoding='utf-8') as f:
            json.dump(contaminated_indices, f, ensure_ascii=False, indent=2)

    # Print match statistics
    print(f"\nAnalysis Summary:")
    print(f"Total RewardBench prompts with matches: {len(best_matches)}")
    if best_matches:
        print(f"Overlap score range: {best_matches[-1]['overlap_score']:.2f} to {best_matches[0]['overlap_score']:.2f}")

    # Process and upload clean datasets one split at a time
    for dataset_name in BASE_DATASETS:
        print(f"\nProcessing {dataset_name}:")
        
        # Load dataset
        pref_dataset = load_dataset(f"esfrankel17/{dataset_name}_binarized_w_weak_preferences")
        
        for split in pref_dataset.keys():
            print(f"\nCleaning split {split}:")
            split_data = pref_dataset[split]
            
            # Get clean indices
            clean_indices = [i for i in range(len(split_data)) if i not in contaminated_indices[dataset_name][split]]
            
            # Create and upload clean split
            clean_split = split_data.select(clean_indices)
            
            print(f"Original size: {len(split_data)}")
            print(f"Contaminated examples: {len(contaminated_indices[dataset_name][split])}")
            print(f"Clean examples: {len(clean_indices)}")
                        
            # Upload clean split directly - no need to convert to dict first
            clean_split.push_to_hub(
                f"esfrankel17/{dataset_name}_binarized_w_weak_preferences_cleaned",
                split=split
            )
            
            # Clear memory
            del split_data, clean_split, clean_indices
        
        # Clear dataset from memory before moving to next one
        del pref_dataset

if __name__ == "__main__":
    main()