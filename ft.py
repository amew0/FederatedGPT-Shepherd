logg = lambda x: f"-------------------------------- {x} ------------------------"
print(logg("STARTED!"))
from time import time
# now
start = time()

import torch
print(torch.cuda.is_available())
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM,  BitsAndBytesConfig
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
datasets.utils.logging.set_verbosity_error()

chavinlo = "chavinlo/alpaca-native"
ME="/dpc/kunf0007/amine"
from datetime import datetime
RUN_ID = datetime.now().strftime("%y%m%d%H")
print(logg(RUN_ID))

chavinlo_tokenizer = LlamaTokenizer.from_pretrained(
    chavinlo,
    cache_dir=f"{ME}/chavinlo/tokenizer",
    padding_side="right",
    pad_token_id=(0))
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    chavinlo,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    cache_dir=f"{ME}/chavinlo/model"
)
print(logg("MODELS_LOADED"))

model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules= ["q_proj","k_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print(logg("LORA'ed"))


prompter = Prompter('alpaca', verbose=False)

cutoff_len = 512
def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["context"],
        data_point["response"],
    )
    tokenized_full_prompt = tokenize(full_prompt, chavinlo_tokenizer)
    return tokenized_full_prompt

from datasets import load_dataset
local_data_path = './data/1/local_training_0.json'
local_output_dir = f'{ME}/output/local_output_{RUN_ID}'
local_data = load_dataset("json", data_files=local_data_path)

local_train_dataset = local_data["train"].shuffle().map(generate_and_tokenize_prompt)
print(logg("DATA_GENERATED_AND_TOKENIZED"))



gradient_accumulation_steps = 8 // 4
def build_local_trainer(
    tokenizer=chavinlo_tokenizer,
    model=model,
    local_micro_batch_size=4,
    gradient_accumulation_steps=2,
    local_learning_rate=3e-4,
    group_by_length=False,
):
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=local_micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1,
        num_train_epochs=30,
        learning_rate=local_learning_rate,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=local_output_dir,
        group_by_length=group_by_length,
        dataloader_drop_last=False,
    )
    local_trainer = SFTTrainer(
        model=model,
        train_dataset=local_train_dataset,
        args=train_args,
        tokenizer=tokenizer,
        dataset_text_field="instruction",
        max_seq_length=512,
    )
    return local_trainer

model = build_local_trainer()

from collections import OrderedDict
import copy
# def initiate_local_training(model=model):
#     model.config.use_cache = False
#     params_dict_new = OrderedDict(
#         (name, param.detach())
#         for name, param in model.named_parameters()
#         if "default" in name
#     )
#     model.state_dict = (
#         lambda instance, *_, **__: get_peft_model_state_dict(
#             instance, params_dict_new, "default"
#         )
#     ).__get__(model, type(model))
#     return model
# model = initiate_local_training()
print(logg("INITIATED_LOCAL_TRAINING"))

import gc

gc.collect()
gc.collect()
model.train()

print(logg("JUST_AFTER_TRAIN_CALL"))


def save(model, tokenizer=chavinlo_tokenizer):
    model.save_pretrained(f"{ME}/model/mylora-shepherd-v{RUN_ID}")
    tokenizer.save_pretrained(f"{ME}/model/mylora-shepherd-v{RUN_ID}")

save(model)
print(logg("SAVED"))

from huggingface_hub import HfFolder

huggingface_token = "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"
HfFolder.save_token(huggingface_token)

model.push_to_hub(f"mylora-shepherd-v{RUN_ID}")
model.push_to_hub(f"mylora-shepherd-v{RUN_ID}")

print(logg("PUSHED"))

end = time()
print(logg(f"ELAPSED - {end - start}"))