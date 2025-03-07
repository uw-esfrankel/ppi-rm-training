import argparse
import json
from pathlib import Path
from random import random
import re
from tqdm import tqdm

from datasets import load_dataset
import jsonlines

from utils.message_utils import generate
from utils.vllm_manager import VLLMManager

USER_PROMPT_SINGLE_TURN_HEADER="""As an evaluation expert, given a question and its two possible answers, please choose which answer best aligns with coherence, accuracy, coverage, and overall quality. Output your judgment in JSON format, where ”rationale” is your explanation, and ”better answer” is an integer type of 1 or 2, for example, {“rationale”: “your explanation”, “better answer”: 1}. Below are the question and its candidate answers:"""

USER_PROMPT_MULTITURN="""As an evaluation expert, given a question and its two possible answers, please choose which answer best aligns with coherence, accuracy, coverage, and overall quality. Output your judgment in JSON format, where ”rationale” is your explanation, and ”better answer” is an integer type of 1 or 2, for example, {“rationale”: “your explanation”, “better answer”: 1}. Below are the question and its candidate answers:

Question: {prompt}
Answer 1: {output_1}
Answer 2: {output_2}"""


def create_single_turn_user_prompt(prompt, output_1, output_2):
    return f"{USER_PROMPT_SINGLE_TURN_HEADER}\n\nQUESTION: {prompt}\n\nANSWER 1: {output_1}\n\nANSWER 2: {output_2}"


def parse_preference(pref_output_str):
    match = re.search(r'\{.*?\}', pref_output_str, re.DOTALL)

    if match:
        try:
            json_content = match.group(0)
            data = json.loads(json_content)
            assert len(data.keys()) == 2 and "rationale" in data and "better answer" in data
            return data, True
        except:
            return None, False
    else:
        return None, False


def get_preference(dataset_name, chosen, option_pair, num_messages, vllm_server_url, model_name):
    system_prompt = ""
    if "nectar" in dataset_name.lower() and num_messages > 2:
        context = chosen[:-2]
        question = chosen[-2]

        user_prompt = USER_PROMPT_MULTITURN.format(
            context=context,
            question=question,
            output_1=option_pair[0],
            output_2=option_pair[1]
        )
    else:
        prompt = chosen[-2]['content']
                
        user_prompt = create_single_turn_user_prompt(prompt, option_pair[0]['content'], option_pair[1]['content'])
        
    pref_output = generate(system_prompt, user_prompt, vllm_server_url, model_name)
    parsed_output, did_parse = parse_preference(pref_output)

    max_retries = 3
    retry_count = 0

    while not did_parse and retry_count < max_retries:
        print(f"WARNING: Did not properly parse the output: {parsed_output}. Retrying... ({retry_count + 1}/{max_retries})")
        
        parsed_output, did_parse = parse_preference(pref_output)
        retry_count += 1

    if retry_count == max_retries:
        print("WARNING: Failed to parse after max retries. Skipped")
        return None

    return parsed_output


def main(args):
    dataset_name = args.dataset_name
    dataset_split = args.dataset_split
    model_name = args.model_name
    seed = args.seed

    dirname = Path(args.dirname) / "elicit_preferences" / args.dataset_name.split("/")[-1] / args.dataset_split / args.model_name.split("/")[-1]
    dirname.mkdir(parents=True, exist_ok=True)

    print(args, file=open(dirname / "args.txt", "w"))

    vllm_manager = VLLMManager(args.model_name, num_gpus=args.num_gpus, port=args.port, seed=args.seed, vllm_process_outdir=dirname)
    vllm_server_url = vllm_manager.vllm_url

    dataset = load_dataset(dataset_name, split=dataset_split).shuffle(seed=seed)
    
    for row in tqdm(dataset):
        chosen, rejected = row["chosen"], row["rejected"]
        num_messages = len(chosen)
        assert num_messages == len(rejected)
        assert num_messages % 2 == 0
        assert chosen[:-1] == rejected[:-1]

        if random() < 0.5:
            option_pair = chosen[-1], rejected[-1]
            chosen_first = True
        else:
            option_pair = rejected[-1], chosen[-1]
            chosen_first = False
        
        preference_info = get_preference(
            dataset_name=dataset_name,
            chosen=chosen,
            option_pair=option_pair, 
            num_messages=num_messages,
            vllm_server_url=vllm_server_url,
            model_name=model_name
        )
        
        if preference_info is None:
            continue

        preference_info["chosen_first"] = chosen_first
        preference_info["orig_chosen"] = chosen
        preference_info["orig_rejected"] = rejected

        with jsonlines.open(f"{dirname}/preferences.jsonl", 'a') as writer:
            writer.write(preference_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

     # vllm arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--write_to_logfile", default=False, action=argparse.BooleanOptionalAction)
    
    # dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, required=True)
    parser.add_argument("--dirname", type=str, default="results")

    args = parser.parse_args()
    main(args)

