from transformers import AutoTokenizer

local_data_path = "./data/1/medical-1-row.json"
ME="/dpc/kunf0097"

name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)


def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    
    tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer)
    print(tokenized_full_prompt)
    return tokenized_full_prompt

from datasets import load_dataset
local_data = load_dataset("json", data_files=local_data_path)



generate_and_tokenize_prompt(local_data["train"][0], tokenizer)