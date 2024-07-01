# %%
import torch
print(torch.cuda.is_available())

# ft-medical.py for evaluation
logg = lambda x: print(f"------------------------ {x} ---------------------------")
import os
import torch
import transformers
import huggingface_hub
import wandb
wandb.require('core')
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from time import time
import gc
import json
# %%
start = time()
load_dotenv()
HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
huggingface_hub.login(token=HF_TOKEN_WRITE)

torch.cuda.empty_cache()

# Set model and tokenizer paths
ME = "/dpc/kunf0097/l3-8b"
RUN_ID = datetime.now().strftime("%y%m%d%H%M%S")
logg(RUN_ID)
logg('llm-as-a-judge.py')


# %%
eval_name = "meta-llama/Meta-Llama-3-8B-Instruct"
eval_tokenizer = AutoTokenizer.from_pretrained(
    eval_name, 
    cache_dir=f"{ME}/tokenizer", 
    pad_token_id=0,
)

eval_model = AutoModelForCausalLM.from_pretrained(
    eval_name,
    cache_dir=f"{ME}/model", 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_buffers=True
)

# %%
# model to be evaluated
name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    name, 
    cache_dir=f"{ME}/tokenizer", 
    pad_token_id=0,
)

model = AutoModelForCausalLM.from_pretrained(
    name,
    cache_dir=f"{ME}/model", 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_buffers=True
)

# %%

# Tokenize prompt function
def tokenize(prompt, tokenizer, add_eos_token=True):
    tokenized = tokenizer(
        prompt,
        return_tensors="pt"
    )
    return tokenized

def generate_and_tokenize_prompt(data_point):
    # for eval, I think, the output part should not be there
    prompt = """<|start_header_id|>system<|end_header_id|> {}<|eot_id|><|start_header_id|>user<|end_header_id|> {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    prompt = prompt.format(data_point["instruction"], data_point["input"])
    tokenized_full_prompt = tokenize(prompt, tokenizer=tokenizer)
    return tokenized_full_prompt

# Load and process dataset
data_path = './data/1/eval_medical_2k.json'
output_dir = f'{ME}/output/output_{RUN_ID}'
data = load_dataset("json", data_files=data_path)
eval_dataset = data["train"].map(generate_and_tokenize_prompt) # not shuffled

# %%
def eval_prompt_tokenizer(generated:str, output:str):
    # we can add the question for context if we fill like
    # [Later] structure (i.e how sentence-structurally the generated output resembled the expected one) and  
    prompt = """<|start_header_id|>system<|end_header_id|>
You are going to act as an LLM evaluator to rate the answer of the medical chatbot on factualness (i.e how contextually the generated output followed the expected reply). Penalize it appropriately for any hallucination, lost of context, or trailing repetition. YOUR RESPONSE IS NOTHING ELSE BUT A FLOAT FROM 0.0 - 5.0 (with format x.x). Where 0.0 indicates the context of the generated response is very far from the expected one. And 5.0 represents otherwise. AGAIN IF YOUR GENERATED ANYTHING ELSE BUT A FLOAT YOU'RE GOING TO CRUSH MY SYSTEM!!<|eot_id|><|start_header_id|>user<|end_header_id|> 
### Expected: {}
### Generated: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    prompt = prompt.format(generated, output)
    tokenized_full_prompt = tokenize(prompt, tokenizer=eval_tokenizer)
    return tokenized_full_prompt

# %%
import re
def extract_score(text):
    match = re.search(r'\b\d+\.\d+\b', text)
    return float(match.group(0)) if match else -1.0

# %%
json_result = f'./out/results_{name[-12:]}_{RUN_ID}.json'
with open(json_result, 'w') as f:
    f.write('[') 
def log2json(res):
    with open(json_result, 'a') as f:
        f.write(',')
        json.dump(res, f, ensure_ascii=False)
        f.flush()

# %%
from tqdm import tqdm
results = []
for i, example in tqdm(enumerate(eval_dataset)):
    res = None
    example["input_ids"] = torch.LongTensor(example["input_ids"]).to(model.device)
    example["attention_mask"] = torch.LongTensor(example["attention_mask"]).to(model.device)

    outputs = model.generate(
        input_ids=example["input_ids"],
        attention_mask = example["attention_mask"],
        max_new_tokens=256,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response_ids = outputs[0][example["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    gt_response = example["output"] # groundtruth

    eval_tokenized = eval_prompt_tokenizer(response, gt_response)

    evals_per_example = 2

    llm_scores = []
    for i in range(evals_per_example):
        # use the evaluator here
        eval_output = eval_model.generate(
            input_ids=torch.LongTensor(eval_tokenized["input_ids"]).to(model.device),
            attention_mask = torch.LongTensor(eval_tokenized["attention_mask"]).to(model.device),
            max_new_tokens=128,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        eval_response = eval_output[0][eval_tokenized["input_ids"].shape[-1]:]
        llm_score = eval_tokenizer.decode(eval_response, skip_special_tokens=True)
        llm_scores.append(extract_score(llm_score))


    res = {
        "output": gt_response,
        "generated": response,
        "llm_scores": llm_scores,
        "avg_llm_score": sum(llm_scores)/len(llm_scores)
    }
    log2json(res)
    
    results.append(res)
    del example
    gc.collect()
    gc.collect()
    # if i == 0: break

end = time()
logg(end-start)