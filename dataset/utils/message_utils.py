import time
import openai
import requests


def parse_out_prompt_turns_hh_format(text):
    prompt_turns = []
    text_split = text.split('Human:')
    for entry in text_split:
        if entry.strip() != '':
            assistant_split = entry.split('Assistant:')
            human_text = assistant_split[0].strip()
            if len(assistant_split) > 1:
                assistant_text = assistant_split[1].strip()
                if human_text:
                    prompt_turns.append({"role": "user", "content": human_text})
                if assistant_text:
                    prompt_turns.append({"role": "assistant", "content": assistant_text})
            else:
                if human_text:
                    prompt_turns.append({"role": "user", "content": human_text})
    return prompt_turns


def generate_using_vllm(system_prompt, prompt, vllm_server_url, model_name, use_system_prompt=True):
    if use_system_prompt:
        system = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }

    message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }
        
    client = openai.OpenAI(
        base_url=f"{vllm_server_url}/v1",
        api_key="ppi"
    )
    if use_system_prompt:
        response = client.chat.completions.create(
            model=model_name,
            messages=[system, message],
            temperature=0.0,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[message],
            temperature=0.0,
        )
    return response.choices[0].message.content
    

def generate(system_prompt, prompt, vllm_server_url, model_name, use_system_prompt=True):
    while True:
        try:
            return generate_using_vllm(system_prompt, prompt, vllm_server_url, model_name, use_system_prompt)
        except requests.RequestException as e:
            print(f"Error during generation from the text generation interface: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.APIError as e:
            print(f"WARNING {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except Exception as e:
            print(f"WARNING: Unexpected exception in generate function {e}")
            return False
