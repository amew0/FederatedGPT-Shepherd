logg = lambda x: f"-------------------------------- {x} ------------------------"
print(logg("STARTED!"))
from time import time
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
import wandb
wandb.require("core") # 

# now
start = time()

# import torch
# print(torch.cuda.is_available())
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, BitsAndBytesConfig
# from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM,  BitsAndBytesConfig
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    # get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
# from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
# import datasets
# from utils.prompter import Prompter
# datasets.utils.logging.set_verbosity_error()

ME="/dpc/kunf0097/l3-8b"
RUN_ID = datetime.now().strftime("%y%m%d%H")
print(logg(RUN_ID))

name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=f"{ME}/l3-8b/tokenizer", 
                                        #   padding_side="right", pad_token_id=(0)
    ) # otherwise causes ValueError: Asking to pad but the tokenizer does not have a padding token. 
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(name, cache_dir=f"{ME}/l3-8b/model",     quantization_config=bnb_config,
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


# prompter = Prompter('alpaca', verbose=False)

def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors=None,
    )
    if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer)
    return tokenized_full_prompt

from datasets import load_dataset
data_path = './data/1/medical.json'
output_dir = f'{ME}/output/output_{RUN_ID}'
data = load_dataset("json", data_files=data_path)

train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)
print(logg("DATA_GENERATED_AND_TOKENIZED"))

gradient_accumulation_steps = 8 // 4
def build_trainer(tokenizer=tokenizer, model=model):
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        num_train_epochs=1,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=output_dir,
        group_by_length=False,
        dataloader_drop_last=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=train_args,
        # dataset_text_field="instruction",
        max_seq_length=512,
    )
    return trainer

trainer = build_trainer()
print(logg("INITIATED_LOCAL_TRAINING"))

import gc

gc.collect()
gc.collect()
trainer.train()

print(logg("JUST_AFTER_TRAIN_CALL"))


# def save(model, tokenizer=tokenizer):
    # tokenizer.save_pretrained(f"{ME}/model/mylora-shepherd-v{RUN_ID}")
    
# save()
# Save model and tokenizer
model_save_path = f"{ME}/model/l3-8b-medical-v{RUN_ID}"
trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(logg("SAVED"))

from huggingface_hub import HfFolder

hf_token_write = os.getenv("HF_TOKEN_WRITE")
if hf_token_write:
    print("write token found!", hf_token_write)
    HfFolder.save_token(hf_token_write)

    # Push model and tokenizer to the hub
    trainer.model.push_to_hub(f"l3-8b-medical-v{RUN_ID}")
    tokenizer.push_to_hub(f"l3-8b-medical-v{RUN_ID}")

print(logg("PUSHED"))
end = time()
print(logg(f"ELAPSED - {end - start}"))