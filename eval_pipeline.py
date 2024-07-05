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


def inspectt(frame):
    logg("")
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(f"\t{arg}: {values[arg]}")
    logg("")


def get_prompts_from_template(filepath, name, eval_name):
    default_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    }
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    candidate_prompt = data[name]["candidate_prompt"]
    evaluator_prompt = data[eval_name]["evaluator_prompt"]
    candidate_generation_config = data[name].get("candidate_generation_config", default_config)
    evaluator_generation_config = data[eval_name].get(
        "evaluator_generation_config", default_config
    )

    print("candidate_prompt: ", candidate_prompt)
    print("evaluator_prompt: ", evaluator_prompt)
    print("candidate_generation_config: ", candidate_generation_config)
    print("evaluator_generation_config: ", evaluator_generation_config)

    return (
        candidate_prompt,
        evaluator_prompt,
        candidate_generation_config,
        evaluator_generation_config,
    )


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
    match = re.search(r"\b\d+(\.\d+)?\b", text)
    return float(match.group(0)) if match else -1.0


def log2json(results, json_result):
    with open(json_result, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def generate_response(model, tokenizer, input_ids, attention_mask, generation_config):
    try:
        output = model.generate(
            input_ids=torch.LongTensor(input_ids).to(model.device),
            attention_mask=torch.LongTensor(attention_mask).to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            **generation_config,
        )
        response_ids = output[0][len(input_ids[0]) :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        return response
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            print(f"Skipping example due to invalid output: {e}")
            return None
        else:
            raise 


def main(
    output_dir=f"./out",
    cache_dir=f"/dpc/kunf0097/l3-8b",
    eval_data_path="./data/1/eval_medical_2k.json",
    log_file=None,
    candidate_name="meta-llama/Meta-Llama-3-8B-Instruct",
    evaluator_name="meta-llama/Meta-Llama-3-8B-Instruct",
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
    candidate_name (str): Model name for evaluation. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    evaluator_name (str): Model name for the evaluator. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    run_id (str): Run ID. Default is current timestamp.
    log2wandb (bool): Whether to log to Weights & Biases. Default is True.
    project (str): WandB project name. Default is huggingface.
    entity (str): WandB entity name. Default is my-ku-org.
    evals_per_example (int): No. of times the example to be evaluated. Default is 2.
    """

    if log2wandb and (project is None or entity is None):
        raise ValueError("Both 'project' and 'entity' must be set if 'log2wandb' is True.")

    if log_file is None:
        log_file = f"{output_dir}/results_{candidate_name.split('/')[1]}_{run_id}.json"

    inspectt(inspect.currentframe())

    (
        candidate_prompt,
        evaluator_prompt,
        candidate_generation_config,
        evaluator_generation_config,
    ) = get_prompts_from_template("template.yaml", candidate_name, evaluator_name)

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)
    torch.cuda.empty_cache()
    logg(run_id)

    evaluator_tokenizer, evaluator_model = get_tokenizer_and_model(
        model_name=evaluator_name, cache_dir=cache_dir
    )

    candidate_tokenizer, candidate_model = get_tokenizer_and_model(
        model_name=candidate_name, cache_dir=cache_dir
    )

    data = load_dataset("json", data_files=eval_data_path)
    eval_dataset = data["train"].map(
        lambda x: generate_and_tokenize_prompt(x, candidate_tokenizer, candidate_prompt)
    )  # not shuffled

    if log2wandb:
        wandb.init(
            project=project,
            entity=entity,
            name=f"laaj-{candidate_name.split('/')[1]}_{run_id}",
        )
        wandb.log({"Evaluation prompt": evaluator_prompt, "Evaluator": evaluator_name})

    results = []
    for i, example in tqdm(enumerate(eval_dataset)):
        res = None
        response = generate_response(
            candidate_model,
            candidate_tokenizer,
            example["input_ids"],
            example["attention_mask"],
            candidate_generation_config,
        )

        if response is None:
            continue
        
        gt_response = example["output"]  # groundtruth
        eval_prompt_tokenized = eval_prompt_tokenizer(
            response, gt_response, evaluator_tokenizer, prompt=evaluator_prompt
        )

        llm_scores = []
        no_scores = []
        for _ in range(evals_per_example):
            generated_score = generate_response(
                evaluator_model,
                evaluator_tokenizer,
                eval_prompt_tokenized["input_ids"],
                eval_prompt_tokenized["attention_mask"],
                evaluator_generation_config,
            )
            if generated_score is None:
                continue
            score = extract_score(generated_score)
            if score >= 0.0 and score <= 5.0:
                llm_scores.append(score)
            else:
                no_scores.append(generated_score)  # to see what evaluator generated
                llm_scores.append(results[i - 1]["running/run_score"] if i > 0 else 0.0)

        res = {
            "expected": gt_response,
            "generated": response,
            "scores": llm_scores,
            "row_avg": sum(llm_scores) / len(llm_scores),
            "no_scores": no_scores,
        }

        results.append(res)

        # Transpose to compute column wise results
        scores_t = list(zip(*[d["scores"] for d in results]))
        row_avg_t = [d["row_avg"] for d in results]

        pcc_results = {
            f"pcc_{i}_{j}": (
                pearsonr(scores_t[i], scores_t[j])[0] if len(scores_t[i]) > 1 else 0
            )
            for i in range(len(scores_t))
            for j in range(i + 1, len(scores_t))
        }  # Calculate PCC for each pair of LLM scores

        column_avg = {
            f"avg_llm_score_{i}": sum(scores) / len(scores)
            for i, scores in enumerate(scores_t)
        }  # Calculate average scores for each set of LLM scores

        run_score = sum(row_avg_t) / len(row_avg_t)

        results[i] = {
            **res,
            "running/pcc": pcc_results,
            "running/column_avg": column_avg,
            "running/run_score": run_score,
        }
        log2json(results, log_file)

        if log2wandb:
            wandb.log(results[i])

        del scores_t
        del example
        gc.collect()
        gc.collect()

    if log2wandb:
        table = wandb.Table(columns=list(results[0].keys()))
        for r in results:
            table.add_data(*r.values())
        wandb.log({"Evaluation Results": table})
        wandb.finish()

    end = time()
    logg(f"Elapsed: {end - start}")


if __name__ == "__main__":
    logg("eval_pipeline.py")
    fire.Fire(main)
