import os
import torch
import transformers
import huggingface_hub
import wandb
from scipy.stats import pearsonr
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import gc
import json
import yaml
import argparse
import re
from tqdm import tqdm
import fire
import inspect


logg = lambda x: print(f"------------------------ {x} ---------------------------")

ME = "/dpc/kunf0097/l3-8b"


def inspectt(frame):
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(f"\t{arg}: {values[arg]}")
    logg("")


def get_prompts_from_template(filepath, model_name):
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data[model_name]["chat_prompt"], data[model_name]["evaluation_prompt"]


def get_tokenizer_and_model(model_name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/tokenizer",
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map="auto",
        offload_buffers=True,
    )
    return tokenizer, model


def tokenize(prompt, tokenizer):
    tokenized = tokenizer(prompt, return_tensors="pt")
    return tokenized


def generate_and_tokenize_prompt(data_point, tokenizer, prompt=None):
    prompt = prompt.format(data_point["instruction"], data_point["input"])
    tokenized_full_prompt = tokenize(prompt, tokenizer=tokenizer)
    return tokenized_full_prompt


def eval_prompt_tokenizer(generated, output, eval_tokenizer, prompt=None):
    prompt = prompt.format(generated, output)
    tokenized_full_prompt = tokenize(prompt, tokenizer=eval_tokenizer)
    return tokenized_full_prompt


def extract_score(text):
    match = re.search(r"\b\d+\.\d+\b", text)
    return float(match.group(0)) if match else -1.0


def log2json(results, json_result):
    with open(json_result, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main(
    output_dir=f"./out",
    cache_dir=f"{ME}",
    eval_data_path="./data/1/eval_medical_2k.json",
    log_file=None,
    name="meta-llama/Meta-Llama-3-8B-Instruct",
    eval_name="meta-llama/Meta-Llama-3-8B-Instruct",
    run_id=datetime.now().strftime("%y%m%d%H%M%S"),
    log2wandb: bool = True,
    project="huggingface",
    entity="my-ku-org",
    evals_per_example=2,
):
    """
    Evaluate a model with LLM-as-a-Judge.

    Args:
    output_dir (str): Directory to save output. Default is './out'.
    cache_dir (str): Directory to load/save tokenizer/model. Default is '/dpc/kunf0097/l3-8b'.
    eval_data_path (str): Path to the evaluation data. Default is './data/1/eval_medical_2k.json'.
    log_file (str): File to dump the outputs of the evaluator. Default is {output_dir}/results_{name.split('/')[1]}_{run_id}.json.
    name (str): Model name for evaluation. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    eval_name (str): Model name for the evaluator. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    run_id (str): Run ID. Default is current timestamp.
    log2wandb (bool): Whether to log to Weights & Biases. Default is True.
    project (str): WandB project name. Default is huggingface.
    entity (str): WandB entity name. Default is my-ku-org.
    evals_per_example (int): No. of times the example to be evaluated. Default is 2.
    """

    chat_prompt, evaluator_prompt = get_prompts_from_template("template.yaml", name)

    if log2wandb and (project is None or entity is None):
        raise ValueError("Both 'project' and 'entity' must be set if 'log2wandb' is True.")

    if log_file is None:
        log_file = f"{output_dir}/results_{name.split('/')[1]}_{run_id}.json"

    inspectt(inspect.currentframe())
    # import sys
    # sys.exit()

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)

    torch.cuda.empty_cache()

    logg(run_id)

    evaluator_tokenizer, evaluator_model = get_tokenizer_and_model(
        model_name=eval_name, cache_dir=cache_dir
    )

    candidate_tokenizer, candidate_model = get_tokenizer_and_model(
        model_name=name, cache_dir=cache_dir
    )

    data = load_dataset("json", data_files=eval_data_path)
    eval_dataset = data["train"].map(
        lambda x: generate_and_tokenize_prompt(x, candidate_tokenizer, chat_prompt)
    )  # not shuffled

    if log2wandb:
        wandb.init(project=project, entity=entity)

    # generation config
    generation_config = {
        "max_new_tokens": 256,
        "eos_token_id": [
            candidate_tokenizer.eos_token_id,
            candidate_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    }

    results = []
    for i, example in tqdm(enumerate(eval_dataset)):
        res = None
        outputs = candidate_model.generate(
            input_ids=torch.LongTensor(example["input_ids"]).to(candidate_model.device),
            attention_mask=torch.LongTensor(example["attention_mask"]).to(
                candidate_model.device
            ),
            **generation_config,
        )

        response_ids = outputs[0][len(example["input_ids"][0]) :]
        response = candidate_tokenizer.decode(response_ids, skip_special_tokens=True)
        gt_response = example["output"]  # groundtruth

        eval_prompt_tokenized = eval_prompt_tokenizer(
            response, gt_response, evaluator_tokenizer, prompt=evaluator_prompt
        )

        llm_scores = []
        for i in range(evals_per_example):
            eval_output = evaluator_model.generate(
                input_ids=torch.LongTensor(eval_prompt_tokenized["input_ids"]).to(
                    candidate_model.device
                ),
                attention_mask=torch.LongTensor(eval_prompt_tokenized["attention_mask"]).to(
                    candidate_model.device
                ),
                **generation_config,
            )

            eval_response = eval_output[0][len(eval_prompt_tokenized["input_ids"][0]) :]
            llm_score = evaluator_tokenizer.decode(eval_response, skip_special_tokens=True)
            llm_scores.append(extract_score(llm_score))

        res = {
            "expected": gt_response,
            "generated": response,
            "llm_scores": llm_scores,
            "avg_llm_score": sum(llm_scores) / len(llm_scores),
        }
        results.append(res)
        log2json(results, log_file)

        if log2wandb:
            wandb_log = {"index": i, "avg_llm_score": res["avg_llm_score"]}
            for j, score in enumerate(llm_scores):
                wandb_log[f"llm_score_{j}"] = score
            wandb.log(wandb_log)

        del example
        gc.collect()
        gc.collect()

    if log2wandb:
        results_t = list(
            zip(*[d["llm_scores"] for d in results])
        )  # Transpose to do PCC easier
        avg_llm_scores = [d["avg_llm_score"] for d in results]

        pcc_results = {
            f"pcc_{i}_{j}": pearsonr(results_t[i], results_t[j])[0]
            for i in range(len(results_t))
            for j in range(i + 1, len(results_t))
        }  # Calculate PCC for each pair of LLM scores

        avg_scores = {
            f"avg_llm_score_{i}": sum(scores) / len(scores)
            for i, scores in enumerate(results_t)
        }  # Calculate average scores for each set of LLM scores

        wandb.log(
            {
                **avg_scores,
                **pcc_results,
                "run_score": sum(avg_llm_scores) / len(avg_llm_scores),
            }
        )  # Log the calculated data to wandb

        wandb.finish()

    end = time()
    logg(end - start)


if __name__ == "__main__":
    logg("eval_pipeline.py")
    fire.Fire(main)
