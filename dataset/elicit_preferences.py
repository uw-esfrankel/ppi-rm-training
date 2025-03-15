import argparse
import json
from pathlib import Path
from random import random
import re
import os

from datasets import (
    load_dataset,
    get_dataset_split_names,
    Dataset,
    concatenate_datasets,
)
from vllm import LLM, SamplingParams

from utils.torch_utils import set_seed_everywhere


USER_PROMPT_SINGLE_TURN_HEADER = """As an evaluation expert, please assess two AI assistant responses to a user's message. Your task is to determine which response better aligns with coherence, accuracy, coverage, and overall quality given the conversation history.

You MUST choose either Response 1 or Response 2 as better - ties are not allowed. Even if the differences are subtle, identify which response is superior overall.

Output your judgment in JSON format where "rationale" is your explanation, and "better_answer" is an integer type of 1 or 2, for example, {"rationale": "your explanation", "better_answer": 1}."""

USER_PROMPT_MULTITURN = """As an evaluation expert, please assess two AI assistant responses to the final user message in a conversation. Your task is to determine which response better aligns with coherence, accuracy, coverage, and overall quality given the conversation history.

You MUST choose either Response 1 or Response 2 as better - ties are not allowed. Even if the differences are subtle, identify which response is superior overall.

Output your judgment in JSON format where "rationale" is your explanation, and "better_answer" is an integer type of 1 or 2, for example, {"rationale": "your explanation", "better_answer": 1}."""


def create_single_turn_user_prompt(user_message, response_1, response_2):
    return (
        USER_PROMPT_SINGLE_TURN_HEADER
        + f"\n\nUSER MESSAGE: {user_message}\n\nRESPONSES TO EVALUATE:\nRESPONSE 1: {response_1}\nRESPONSE 2: {response_2}"
    )


def create_multiturn_user_prompt(
    conversation_history, user_message, response_1, response_2
):
    return (
        USER_PROMPT_MULTITURN
        + f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n\nFINAL USER MESSAGE: {user_message}\n\nRESPONSES TO EVALUATE:\nRESPONSE 1: {response_1}\nRESPONSE 2: {response_2}"
    )


def format_conversation_history(messages):
    """
    Format a list of messages into a conversation history string.

    Args:
        messages: List of message dictionaries, where each message has:
            - 'role': Either 'human' or 'assistant'
            - 'content': The message text

    Returns:
        A formatted string representing the conversation history
    """
    formatted_history = []

    for message in messages:
        role = message["role"].upper()
        content = message["content"].strip()

        # Format each message as "ROLE: Content"
        formatted_message = f"{role}: {content}"
        formatted_history.append(formatted_message)

    return "\n".join(formatted_history)


def bespoke_extract_values(text):
    """
    Extract values from a text string with key-value pairs.
    Handles double quotes within values.

    Args:
        text (str): Text string in the format '"key": "value"'

    Returns:
        dict: Dictionary with extracted key-value pairs
    """
    result = {}
    i = 0

    def skip_whitespace():
        nonlocal i
        while i < len(text) and text[i].isspace():
            i += 1

    def parse_quoted_string():
        nonlocal i
        value = ""
        i += 1  # Skip opening quote

        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                # Handle escape sequences
                i += 1
                if text[i] == '"':
                    value += '"'
                else:
                    value += "\\" + text[i]
            elif text[i] == '"':
                # End of quoted string
                i += 1
                return value
            else:
                value += text[i]
            i += 1

        return value

    while i < len(text):
        skip_whitespace()

        # Check if we've reached the end
        if i >= len(text):
            break

        # We expect a key, which should be a quoted string
        if text[i] != '"':
            i += 1
            continue  # Skip non-quoted content

        key = parse_quoted_string()

        skip_whitespace()

        # We expect a colon after the key
        if i >= len(text) or text[i] != ":":
            i += 1
            continue
        i += 1  # Skip the colon

        skip_whitespace()

        # We expect a value, which could be a quoted string or a number
        if i < len(text) and text[i] == '"':
            value = parse_quoted_string()
            result[key] = value
        elif i < len(text) and text[i].isdigit():
            # Parse a number
            num_str = ""
            while i < len(text) and text[i].isdigit():
                num_str += text[i]
                i += 1
            result[key] = int(num_str)
        else:
            i += 1

        # Skip to the next key-value pair
        while i < len(text) and (i >= len(text) or text[i] != '"'):
            if i < len(text):
                i += 1
            else:
                break

    return result


def parse_preference(pref_output_str):
    match = re.search(r"\{.*?\}", pref_output_str, re.DOTALL)

    if match:
        try:
            json_content = match.group(0)
            data = json.loads(json_content)
            assert (
                len(data.keys()) == 2
                and "rationale" in data
                and "better_answer" in data
                and int(data["better_answer"]) in [1, 2]
            )
            return data
        except Exception as e:
            print(e)
            try:
                data = bespoke_extract_values(pref_output_str)
                assert (
                    len(data.keys()) == 2
                    and "rationale" in data
                    and "better_answer" in data
                    and int(data["better_answer"]) in [1, 2]
                )
                return data
            except Exception as e:
                print(e)
                try:
                    # only try to parse better_answer as 1 or 2 from the pref_output_str
                    better_answer_match = re.search(
                        r'"better_answer":\s*([12])', pref_output_str
                    )
                    if better_answer_match:
                        return {
                            "rationale": "Failed to parse full rationale",
                            "better_answer": int(better_answer_match.group(1)),
                        }
                    else:
                        print(
                            f"WARNING: Did not find better_answer in the output: {pref_output_str}. Skipping..."
                        )
                        return None
                except Exception as e:
                    print(
                        f"WARNING: Did not properly parse the output: {pref_output_str}. Skipping..."
                    )
                    print(e)
                    return None
    else:
        print(
            f"WARNING: Did not find a JSON object in the output: {pref_output_str}. Skipping..."
        )
        return None


def create_user_prompt(chosen, option_pair, dataset_name, num_messages):
    if "nectar" in dataset_name.lower() and num_messages > 2:
        return create_multiturn_user_prompt(
            conversation_history=format_conversation_history(chosen[:-2]),
            user_message=chosen[-2]["content"],
            response_1=option_pair[0]["content"],
            response_2=option_pair[1]["content"],
        )
    else:
        return create_single_turn_user_prompt(
            user_message=chosen[-2]["content"],
            response_1=option_pair[0]["content"],
            response_2=option_pair[1]["content"],
        )


def process_dataset(dataset, dataset_name, seed, llm):
    chosen_list, rejected_list = dataset["chosen"], dataset["rejected"]
    chosen_is_first_list = [random() < 0.5 for _ in range(len(chosen_list))]
    option_pair_list = [
        (chosen_list[i][-1], rejected_list[i][-1])
        if chosen_is_first_list[i]
        else (rejected_list[i][-1], chosen_list[i][-1])
        for i in range(len(chosen_list))
    ]
    user_prompt_list = [
        create_user_prompt(
            chosen_list[i],
            option_pair_list[i],
            dataset_name,
            num_messages=len(chosen_list[i]),
        )
        for i in range(len(chosen_list))
    ]
    messages_list = [
        [{"role": "user", "content": user_prompt_list[i]}]
        for i in range(len(user_prompt_list))
    ]
    assert (
        len(messages_list)
        == len(chosen_list)
        == len(rejected_list)
        == len(option_pair_list)
        == len(user_prompt_list)
    )

    params = SamplingParams(temperature=0.0, seed=seed, max_tokens=1024)

    outputs = llm.chat(messages_list, sampling_params=params, use_tqdm=True)

    parsed_outputs = [parse_preference(output.outputs[0].text) for output in outputs]
    indices_to_keep_parsed = [
        i for i in range(len(parsed_outputs)) if parsed_outputs[i] is not None
    ]
    # only keep the indices where the two responses are different and better_answer is 1 or 2
    indices_to_keep_distinct = [
        i
        for i in indices_to_keep_parsed  # Only iterate over valid indices
        if chosen_list[i][-1]["content"] != rejected_list[i][-1]["content"]
        and parsed_outputs[i]["better_answer"] in [1, 2]
    ]
    indices_to_keep = sorted(indices_to_keep_distinct)  # No need for set intersection

    parsed_outputs_to_keep = [parsed_outputs[i] for i in indices_to_keep]
    chosen_list_to_keep = [chosen_list[i] for i in indices_to_keep]
    rejected_list_to_keep = [rejected_list[i] for i in indices_to_keep]
    chosen_is_first_list_to_keep = [chosen_is_first_list[i] for i in indices_to_keep]
    original_chosen_rating_list_to_keep = [
        dataset[i]["chosen_rating"] for i in indices_to_keep
    ]
    original_rejected_rating_list_to_keep = [
        dataset[i]["rejected_rating"] for i in indices_to_keep
    ]

    assert (
        len(parsed_outputs_to_keep)
        == len(chosen_list_to_keep)
        == len(rejected_list_to_keep)
        == len(chosen_is_first_list_to_keep)
    )
    assert all(
        parsed_output["better_answer"] in [1, 2]
        for parsed_output in parsed_outputs_to_keep
    )

    # check whether the model agreed with the original choice
    model_agreed_with_original_list = [
        parsed_output["better_answer"] == 1
        if chosen_is_first_list_to_keep[i]
        else parsed_output["better_answer"] == 2
        for i, parsed_output in enumerate(parsed_outputs_to_keep)
    ]

    # choose the model's choice
    model_chosen_list = [
        chosen_list_to_keep[i]
        if model_agreed_with_original_list[i]
        else rejected_list_to_keep[i]
        for i in range(len(chosen_list_to_keep))
    ]
    model_rejected_list = [
        rejected_list_to_keep[i]
        if model_agreed_with_original_list[i]
        else chosen_list_to_keep[i]
        for i in range(len(chosen_list_to_keep))
    ]

    # Add debug assertions before the main check
    assert len(model_chosen_list) == len(chosen_list_to_keep), (
        "Lists must be same length"
    )
    assert len(model_agreed_with_original_list) == len(chosen_list_to_keep), (
        "Agreement list must match length"
    )

    # Print problematic indices for debugging
    for i in range(len(chosen_list_to_keep)):
        if (
            not model_agreed_with_original_list[i]
            and model_chosen_list[i] == chosen_list_to_keep[i]
        ):
            print(f"Inconsistency found at index {i}:")
            print(f"  model_chosen: {model_chosen_list[i]}")
            print(f"  chosen_to_keep: {chosen_list_to_keep[i]}")
            print(f"  model_agreed: {model_agreed_with_original_list[i]}")

    # Break down the original assertion into steps
    disagreement_indices = [
        i
        for i in range(len(chosen_list_to_keep))
        if not model_agreed_with_original_list[i]
    ]

    for idx in disagreement_indices:
        assert model_chosen_list[idx] != chosen_list_to_keep[idx], (
            f"Model choice should differ from original when disagreement is marked. "
            f"Index {idx}: model_chosen={model_chosen_list[idx]}, "
            f"chosen_keep={chosen_list_to_keep[idx]}"
        )

    # Original assertion remains as final check
    assert all(
        model_chosen_list[i] != chosen_list_to_keep[i]
        for i in range(len(chosen_list_to_keep))
        if not model_agreed_with_original_list[i]
    )

    return {
        "original_chosen": chosen_list_to_keep,
        "original_chosen_rating": original_chosen_rating_list_to_keep,
        "original_rejected": rejected_list_to_keep,
        "original_rejected_rating": original_rejected_rating_list_to_keep,
        "model_chosen": model_chosen_list,
        "model_rejected": model_rejected_list,
        "model_agreed_with_original": model_agreed_with_original_list,
    }


def main(args):
    dataset_name = args.dataset_name
    model_name = args.model_name
    seed = args.seed

    # Get SLURM array task ID and total tasks
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    total_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    dataset_name_split = dataset_name.split("/")[-1]
    model_name_split = model_name.split("/")[-1]

    dataset_split_names = get_dataset_split_names(dataset_name)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.num_gpus,
        seed=seed,
        gpu_memory_utilization=0.95,
    )

    for dataset_split in dataset_split_names:
        # Update dirname to include task ID
        base_dir = (
            Path(args.dirname)
            / "elicit_preferences"
            / dataset_name_split
            / dataset_split
            / model_name_split
        )
        dirname = base_dir / f"task_{task_id}"
        dirname.mkdir(parents=True, exist_ok=True)

        print(args, file=open(dirname / "args.txt", "w"))

        # Check if the results file already exists; if it does, skip this split
        if (dirname / "results.json").exists():
            print(f"Skipping split {dataset_split} because results file already exists")
            continue

        # Load and shuffle the full dataset
        full_dataset = load_dataset(dataset_name, split=f"{dataset_split}").shuffle(
            seed=seed
        )

        # Calculate chunk size and get this task's portion
        chunk_size = len(full_dataset) // total_tasks
        start_idx = task_id * chunk_size
        end_idx = (
            start_idx + chunk_size if task_id < total_tasks - 1 else len(full_dataset)
        )
        dataset = full_dataset.select(range(start_idx, end_idx))

        dataset_dict = process_dataset(dataset, dataset_name, seed, llm)

        new_preference_dataset = Dataset.from_dict(dataset_dict)

        # Save the task results locally
        task_results_path = dirname / "results.json"
        new_preference_dataset.save_to_disk(task_results_path)
        print(f"Saved task {task_id} results to {task_results_path}")

        # Check if all tasks have completed
        task_dirs = list(base_dir.glob("task_*"))
        completed_tasks = [d for d in task_dirs if (d / "results.json").exists()]

        if len(completed_tasks) == total_tasks:
            print("All tasks completed. Combining results and pushing to hub...")
            # Combine all task results
            all_results = []
            for task_dir in completed_tasks:
                try:
                    task_dataset = Dataset.load_from_disk(task_dir / "results.json")
                    all_results.append(task_dataset)
                except Exception as e:
                    print(f"Error loading {task_dir}: {e}")
                    continue

            if all_results:
                combined_dataset = concatenate_datasets(all_results)
                # Push the combined dataset to hub
                combined_dataset.push_to_hub(
                    repo_id=f"esfrankel17/{dataset_name_split}_{model_name_split}_preferences",
                    split=dataset_split,
                )
                print(
                    f"Pushed combined {dataset_name_split}_{model_name_split}_preferences to the hub"
                )
            else:
                print("Error: No valid results to combine")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    # vllm arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--write_to_logfile", default=False, action=argparse.BooleanOptionalAction
    )

    # dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dirname", type=str, default="results")

    args = parser.parse_args()

    set_seed_everywhere(args.seed)

    main(args)
