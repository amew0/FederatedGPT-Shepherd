import os
import torch
import transformers
import huggingface_hub
import wandb
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import gc
import json
import argparse
import re
from tqdm import tqdm
import fire
import inspect


logg = lambda x: print(
    f"------------------------ {x} ---------------------------"
)

ME = "/dpc/kunf0097/l3-8b"

def inspectt(frame):
    args, _, _, values = inspect.getargvalues(frame)
    logg("Args passed:")
    for arg in args:
        print(f"\t{arg}: {values[arg]}")
    logg("")


def tokenize(prompt, tokenizer):
    tokenized = tokenizer(prompt, return_tensors="pt")
    return tokenized


def generate_and_tokenize_prompt(data_point, tokenizer):
    prompt = """<|start_header_id|>system<|end_header_id|> {}<|eot_id|><|start_header_id|>user<|end_header_id|> {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    prompt = prompt.format(data_point["instruction"], data_point["input"])
    tokenized_full_prompt = tokenize(prompt, tokenizer=tokenizer)
    return tokenized_full_prompt


def eval_prompt_tokenizer(generated, output, eval_tokenizer):
    prompt = """<|start_header_id|>system<|end_header_id|>
You are going to act as an LLM evaluator to rate the answer of the medical chatbot on factualness (i.e how contextually the generated output followed the expected reply). Penalize it appropriately for any hallucination, lost of context, or trailing repetition. YOUR RESPONSE IS NOTHING ELSE BUT A FLOAT FROM 0.0 - 5.0 (with format x.x). Where 0.0 indicates the context of the generated response is very far from the expected one. And 5.0 represents otherwise. AGAIN IF YOUR GENERATED ANYTHING ELSE BUT A FLOAT YOU'RE GOING TO CRUSH MY SYSTEM!!<|eot_id|><|start_header_id|>user<|end_header_id|> 
### Expected: {}
### Generated: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    prompt = prompt.format(generated, output)
    tokenized_full_prompt = tokenize(prompt, tokenizer=eval_tokenizer)
    return tokenized_full_prompt


def extract_score(text):
    match = re.search(r"\b\d+\.\d+\b", text)
    return float(match.group(0)) if match else -1.0


def log2json(results, json_result):
    # Write the updated results back to the JSON file
    with open(json_result, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main(
    data_path="./data/1/eval_medical_2k.json",
    output_dir=f"./out",
    cache_dir=f"{ME}",
    name="meta-llama/Meta-Llama-3-8B-Instruct",
    eval_name="meta-llama/Meta-Llama-3-8B-Instruct",
    run_id=datetime.now().strftime("%y%m%d%H%M%S"),
):
    """
    Evaluate a model with LLM-as-a-Judge.

    Args:
    data_path (str): Path to the evaluation data. Default is './data/1/eval_medical_2k.json'.
    output_dir (str): Directory to save output. Default is './out'.
    cache_dir (str): Directory to load/save tokenizer/model. Default is '/dpc/kunf0097/l3-8b'.
    name (str): Model name for evaluation. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    eval_name (str): Model name for the evaluator. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    run_id (str): Run ID. Default is current timestamp.
    """

    # log args
    inspectt(inspect.currentframe())

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)

    torch.cuda.empty_cache()

    logg(run_id)
    # import sys

    # sys.exit()

    eval_tokenizer = AutoTokenizer.from_pretrained(
        eval_name,
        cache_dir=f"{cache_dir}/tokenizer",
        pad_token_id=0,
    )

    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map="auto",
        offload_buffers=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        cache_dir=f"{cache_dir}/tokenizer",
        pad_token_id=0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map="auto",
        offload_buffers=True,
    )

    data = load_dataset("json", data_files=data_path)
    eval_dataset = data["train"].map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer)
    )  # not shuffled

    json_result = f"{output_dir}/results_{name[-12:]}_{run_id}.json"

    results = []
    for i, example in tqdm(enumerate(eval_dataset)):
        res = None
        example["input_ids"] = torch.LongTensor(example["input_ids"]).to(
            model.device
        )
        example["attention_mask"] = torch.LongTensor(
            example["attention_mask"]
        ).to(model.device)

        outputs = model.generate(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"],
            max_new_tokens=256,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(""),
            ],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response_ids = outputs[0][example["input_ids"].shape[-1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        gt_response = example["output"]  # groundtruth

        eval_tokenized = eval_prompt_tokenizer(
            response, gt_response, eval_tokenizer
        )

        evals_per_example = 2

        llm_scores = []
        for i in range(evals_per_example):
            eval_output = eval_model.generate(
                input_ids=torch.LongTensor(eval_tokenized["input_ids"]).to(
                    model.device
                ),
                attention_mask=torch.LongTensor(
                    eval_tokenized["attention_mask"]
                ).to(model.device),
                max_new_tokens=128,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids(""),
                ],
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            eval_response = eval_output[0][
                eval_tokenized["input_ids"].shape[-1] :
            ]
            llm_score = eval_tokenizer.decode(
                eval_response, skip_special_tokens=True
            )
            llm_scores.append(extract_score(llm_score))

        res = {
            "output": gt_response,
            "generated": response,
            "llm_scores": llm_scores,
            "avg_llm_score": sum(llm_scores) / len(llm_scores),
        }
        results.append(res)
        log2json(results, json_result)

        del example
        gc.collect()
        gc.collect()

    end = time()
    logg(end - start)


if __name__ == "__main__":
    logg("eval_pipeline.py")
    fire.Fire(main)
