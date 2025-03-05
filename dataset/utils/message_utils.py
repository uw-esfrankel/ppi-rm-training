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