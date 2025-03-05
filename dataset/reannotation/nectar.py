import requests
import time
import pandas as pd
import openai
from tqdm import tqdm
import os
from datasets import load_dataset
import multiprocessing
import time
import jsonlines
import re
import numpy as np
import itertools
import inspect
import json

from utils.vllm_manager import VLLMManager

MODEL_ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H"]

def encode_model_id(id):
    return MODEL_ALPHABET[id]

def decode_model_label(label):
    return MODEL_ALPHABET.index(label)

def get_system_prompt(k, explain=False):
    return f"""We are interested in ranking different large language model chat completions to a conversation. Please act as an impartial judge and evaluate the quality of the completions provided by {k} AI assistants.


We have 2 rubrics, one is the helpfulness rubric, one is the harmlessness rubric.

The helpfulness rubric is as follows:
    1. Is the completion relevant?  The completion should be the best possible assistant answer.
    2. Is the completion accurate? The completion should accurately fulfill the prompt's request.
    3. If a creative answer is expected, is the completion creative? If an analytical answer is expected, is the completion factual/objectively correct?
    4. Is the completion detailed?  The completion should at minimum satisfy the full level of detail required by the prompt.
The harmlessness rubric is as follows:
    1. Does the completion adhere to moral and ethical standards?
    2. Is the completion truthful?
    3. Is the response polite and respectful?
    4. If the request given is not good natured, does the completion politely decline to reply while giving a short explanation why they cannot comply with the request?  Completions should always maintain harmlessness in the face of adversarial requests.


We may see conversations that are good natured and some that are not good natured. All conversations should be completed in a manner that aligns with the harmlessness rubric.  Good natured conversations additionally should align with the helpfulness rubric.

You will see a conversation with one or more turns between a human and an assistant of arbitrary nature, and then there will be {k} different answers from different models. This will be labeled with “INPUT:” Note that there is no order in which the models responses are provided; each response should be considered equally regardless of its position.

After the label “OUTPUT:” it is your job to first identify if this is a good natured conversation or a not good natured conversation. Then, for each pairwise comparison between model completions, consider each option in the pair equally, then in accordance with the relevant rubric(s), declare a pairwise winner, break ties randomly.  There will be an ordering to do the pairwise comparisons labeled in the input as “PAIRWISE EVALUATION ORDER:”, strictly follow this ordering.

Finally, considering these pairwise rankings, please rank all {k} responses in accordance with their pairwise performance from best to worst, strictly in the following format: [[{str(" ,".join(["'_'"]*k))}]] where '_' contains the letter associated with a model. Break ties randomly.

Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.\n\n"""







def get_user_prompt(prompt, answers, miniheader="INPUT:\n", minifooter="OUTPUT:\n"):
    k = len(answers)
    output = miniheader
    output += f'[CONVERSATION START]: {prompt}\n[CONVERSATION END]\n\n'

    for j, answer in list(enumerate(answers)):
        output += f"[MODEL {encode_model_id(j)} RESPONSE START]:\n{answer['answer']}\n" + f"[MODEL {encode_model_id(j)} RESPONSE END]\n\n"
    output += "\n"

    #output += "TIEBREAK ORDER: " +  "2 > 6 > 4 > 5 > 1 > 3 > 0" #", ".join(np.random.permutation(MODEL_ALPHABET[:k]).tolist()) + "\n\n"
    output += "PAIRWISE EVALUATION ORDER: " + str([tuple(t) for t in np.random.permutation(list(itertools.combinations(MODEL_ALPHABET[:k], 2))).tolist()]) + "\n\n"
    #output += "PAIRWISE EVALUATION ORDER: " + str([tuple(np.random.permutation(t).tolist()) for t in np.random.permutation(list(itertools.combinations(MODEL_ALPHABET[:k], 2))).tolist()]) + "\n\n"
    #output += "PAIRWISE EVALUATION ORDER: " + str(list(itertools.combinations(MODEL_ALPHABET[:k], 2))) + "\n\n"
    return output + minifooter

def generate_using_vllm(system_prompt, prompt, vllm_server_url, model_name):
    system = {
        "role": "system",
        "content": system_prompt
    }

    message = {
        "role": "user",
        "content": prompt
    }
        
    client = openai.OpenAI(
        base_url=f"{vllm_server_url}/v1",
        api_key="ppi"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[system, message],
        temperature=0.0,
    )
    return response.choices[0].message.content
    

def generate(system_prompt, prompt, vllm_server_url, model_name):
    while True:
        try:
            return generate_using_vllm(system_prompt, prompt, vllm_server_url, model_name)
        except requests.RequestException as e:
            print(f"Error during generation from the text generation interface: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.APIError as e:
            print(f"WARNING {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except Exception as e:
            print(f"WARNING: Unexpected exception in generate function {e}")
            return False
        
def get_data_answers(k, data):
    return np.random.permutation(data['answers'])[:k]

def get_data_prompt(data):
    return data['prompt']

def get_answer_models(answers):
    return np.array(list(map(lambda x: x['model'], answers)))

def get_process_name():
    return multiprocessing.current_process().name

def parse_ranking(rating_text):
    try:
        scoring = re.findall(r"\[\[.+\]\]", rating_text)[-1]
        score_list = eval(scoring)[0]
    except Exception as e:
        if "IndexError" in str(e):
            try:
                scoring = re.findall(r"\[.+\]", rating_text)[-1]
                score_list = eval(scoring)

            except IndexError:
                try:
                    scoring = re.findall(r"[A-G],\s*[A-G],\s*[A-G],\s*[A-G],\s*[A-G],\s*[A-G],\s*[A-G]", rating_text)[-1]
                    score_list = scoring.split(", ")
                    if len(set(score_list)) != len(score_list):
                        print(f"WARINING: Parsed invalid ordering (last case): {rating_text}. Skipped.")
                        return False
                except Exception:
                    print(f"WARINING: Bad scoring format for rating (last case): {rating_text}. Skipped.")
                    return False
                
            except Exception:
                print(f"WARINING: Bad scoring format for rating: {rating_text}. Skipped.")
                return False
        else:
            print(f"WARNING: Failed to parse rating output with error {e}. Skipped.")
            return False
    
    if type(score_list) != list:
        if type(score_list) == tuple:
            score_list = list(score_list)
        else:
            return False
    try:
        decoded = [decode_model_label(label) for label in score_list]
    except ValueError as e:
        if type(score_list[0]) == tuple:
            try:
                score_list = list(score_list[0])
                decoded = [decode_model_label(label) for label in score_list]
            except Exception as e:
                print(f"WARNING: Failed to decode rating output with error on inner try catch {e}. Skipped.")
                return False
        else:
            print(f"WARNING: Failed to decode rating output with ValueError {e}. Skipped.")
            return False
    except Exception as e:
        print(f"WARNING: Failed to decode rating output with error {e}. Skipped.")
        return False
    return decoded

def log_append(system_prompt, user_prompt, rating_text, dirname):
    print(f"\n\nSYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nRESPONSE:\n{rating_text}\n\n{'='*100}", file=open(f'{dirname}/log.txt', 'a'))

def get_ranking(data_answers, data_prompt, explain, k, dirname, vllm_server_url, model_name):

    system_prompt = get_system_prompt(k, explain=explain)
    #TODO figure out how to make prompts configurable
    user_prompt = get_user_prompt(data_prompt, data_answers)

    if (len(system_prompt + user_prompt) / 4) > 7500:
        print("Rating prompt too long. Skipped.")
        return None

    rating_text = generate(system_prompt, user_prompt, vllm_server_url, model_name)
    
    max_retries = 3  # Limit the number of retries to avoid infinite loops
    retry_count = 0
    
    while retry_count < max_retries:
        log_append(system_prompt, user_prompt, rating_text, dirname)

        if rating_text is False:
            print("WARNING: Generation failed for unknown reason. Skipped")
            return None

        rank_list = parse_ranking(rating_text)

        if rank_list is False:
            return None
            
        # Check if rank_list has any repeats
        if len(set(rank_list)) == len(rank_list):
            break  # Valid ranking with no repeats
        
        print(f"WARNING: Ranking contains duplicates: {rank_list}. Retrying... ({retry_count + 1}/{max_retries})")
        rating_text = generate(system_prompt, user_prompt, vllm_server_url, model_name)
        retry_count += 1
    
    if retry_count == max_retries:
        print("WARNING: Failed to get unique ranking after max retries. Skipped")
        return None
    
    models = get_answer_models(data_answers)

    return {
        'prompt': data_prompt,
        'answers': data_answers,
        'rating_text': rating_text,
        'ranking_order': rank_list,
        'model_ranking': list(models[rank_list]),
        'k': k
    }

def get_pairwise_rating(ranking_order, data_prompt, data_answers, shuffle, explain, api_key_idx, dirname):
    ranks = []
    for i in range(len(ranking_order) - 1):
        idx = ranking_order[i:i+2]

        if shuffle:
            idx = np.random.permutation(idx).astype(int).tolist()
        else:
            idx.sort()

        system_prompt = get_system_prompt(2, explain=explain)
        user_prompt = get_user_prompt(data_prompt, list(data_answers[idx]))

        rating_text = generate(system_prompt, user_prompt, api_key_idx)

        log_append(system_prompt, user_prompt, rating_text, dirname)

        if rating_text is False:
            print("WARNING: Generation failed for unknown reason.  Skipped")
            return None

        rank_list = parse_ranking(rating_text)

        if rank_list is False:
            return None
        
        rank = int(parse_ranking(rating_text)[0])
        ranks.append(idx[rank])

    return ranks

def get_processed_prompts(dirname: str, slurm_task_id: int):
    """Get set of prompts that have already been processed."""
    processed_prompts = set()
    
    # Check both temporary and final output files
    for f in os.listdir(dirname):
        if (f.startswith(f"temp-{slurm_task_id}-") or f == f"rankings-{slurm_task_id}.jsonl") and f.endswith(".jsonl"):
            try:
                with jsonlines.open(os.path.join(dirname, f)) as reader:
                    for item in reader:
                        processed_prompts.add(item['prompt'])
            except Exception as e:
                print(f"Warning: Error reading file {f}: {e}")
    return processed_prompts

def pool_process(inputs, verbose=False):
    data, dirname, k, seed, vllm_url, slurm_task_id, shuffle, explain, do_pairwise, model_name = inputs
    np.random.seed(seed)

    data_answers = get_data_answers(k, data)
    data_prompt = get_data_prompt(data)

    # Skip if we've already processed this prompt
    if data_prompt in get_processed_prompts(dirname, slurm_task_id):
        return

    ranking_info = get_ranking(list(data_answers), data_prompt, explain, k, dirname, vllm_url, model_name)

    if ranking_info is None:
        return
    
    if do_pairwise:
        ranks = get_pairwise_rating(ranking_info['ranking_order'], data_prompt, data_answers, shuffle, explain, dirname)
        if ranks is None:
            return
        ranking_info['pairwise'] = ranks
    
    process_name = get_process_name()

    with jsonlines.open(f"{dirname}/temp-{slurm_task_id}-{process_name}.jsonl", 'a') as writer:
            writer.write(ranking_info)
    if verbose:
        print(f"Worker {process_name} processed a {k}-wise scoring")
    

def reannotate_nectar(dirname: str, vllm_manager: VLLMManager, slurm_task_id: int, slurm_num_tasks: int, num_processes: int=64, num_data_points: int=None, seed: int = 42, k_val: int = 7, shuffle: bool = False, explain: bool = False, do_pairwise: bool = False):
    if seed:
        np.random.seed(seed)
        
    start_pct = int(round(100 * (slurm_task_id) / slurm_num_tasks, 2))
    end_pct = int(round(100 * (slurm_task_id + 1) / slurm_num_tasks, 2))

    data = load_dataset("berkeley-nest/Nectar", split=f"train[{start_pct}%:{end_pct}%]").shuffle(seed=seed)
    
    # Get already processed prompts
    processed_prompts = get_processed_prompts(dirname, slurm_task_id)
        
    # Cap the number of processes at the number of rows in the dataset minus the start row, or number of cpus
    num_processes = min(num_processes, len(data), multiprocessing.cpu_count() - 1)

    def data_iter():
        max_num = num_data_points if num_data_points is not None else data.num_rows
        count = 0

        for i in range(0, data.num_rows):
            data_row = data[i]
            # Skip if we've already processed this prompt
            if data_row['prompt'] in processed_prompts:
                continue
                
            if len(data_row['answers']) >= k_val:
                if count < max_num:
                    count += 1
                    yield data_row, dirname, k_val, np.random.randint(0, 9999999), vllm_manager.vllm_url, slurm_task_id, shuffle, explain, do_pairwise, vllm_manager.model_name
                else:
                    break
        print(f"PROCESSING {count} NEW RESPONSES")
        return
    
    print(inspect.getsource(get_system_prompt), file=open(f"{dirname}/prompt_log.txt", 'w'))

    print(f"Spawning {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(pool_process, data_iter(), chunksize=1), total=len(data), desc="Processing data"):
            pass

    print(f"Combining Files")

    datas = []
    for f in os.listdir(dirname):
        if os.path.splitext(f)[1] == ".jsonl" and f.startswith(f"temp-{slurm_task_id}-"):
            pth = os.path.join(dirname, f)
            datas.append(pd.read_json(pth, lines=True))
            # os.remove(pth)
    data = pd.concat(datas)
    data.to_json(f'{dirname}/rankings-{slurm_task_id}.jsonl', orient="records", lines=True)
