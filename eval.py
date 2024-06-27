# ft-medical.py for evaluation
logg = lambda x: print(f"------------------------ {x} ---------------------------")
import os
import gc
import torch
import transformers
import huggingface_hub
import wandb
wandb.require('core')
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from time import time

cudas = os.getenv('CUDA_VISIBLE_DEVICES')
logg(cudas)
if cudas: 
    gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] 
else: gpus = [torch.device('cuda')] 

gpu0, gpu1 = gpus[0], None
if len(gpus) > 1: gpu1 = gpus[1]

start = time()

# Load environment variables
load_dotenv()

# Hugging Face token
HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")

# Login to Hugging Face Hub
huggingface_hub.login(token=HF_TOKEN_WRITE)


torch.cuda.empty_cache()

# Set model and tokenizer paths
ME = "/dpc/kunf0097/l3-8b"
RUN_ID = datetime.now().strftime("%y%m%d%H%M%S")
logg(RUN_ID)
logg('eval.py')

# Initialize tokenizer
name = "amew0/l3-8b-medical-v240623023136"

# Initialize model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    name, 
    cache_dir=f"{ME}/model",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    return_dict=True
)

tokenizer = AutoTokenizer.from_pretrained(
    name, 
    cache_dir=f"{ME}/tokenizer", 
    padding_side="right", 
    pad_token_id=(0),
    legacy=False
)
tokenizer.pad_token = tokenizer.eos_token


# Prepare model for LoRA training
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Tokenize prompt function
cutoff_len=128 # most important hp to control CUDA OOM
def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,  # Use the cutoff_len variable you've defined
        padding="max_length",  # Ensure padding is done to the max_length
        return_tensors="pt",   # Return PyTorch tensors
    )
    result["input_ids"] = result["input_ids"].flatten()
    result["attention_mask"] = result["attention_mask"].flatten()

    if add_eos_token and result["input_ids"].shape[0] < cutoff_len:
        # Append eos_token_id to each sequence in the batch
        result["input_ids"][-1] = tokenizer.eos_token_id
        result["attention_mask"][-1] = 1

    result["labels"] = result["input_ids"].clone()  # Clone input_ids for labels
    return result

def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer).to('cuda:0')
    # if gpu1: tokenized_full_prompt.to(gpu1)
    return tokenized_full_prompt

# Load and process dataset
data_path = './data/1/eval_medical_2k.json'
output_dir = f'{ME}/output/output_{RUN_ID}'
data = load_dataset("json", data_files=data_path)
train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)

from sklearn.metrics import  f1_score

def compute_metrics(pred):
    squad_labels = pred.label_ids.flatten()
    squad_preds = pred.predictions.argmax(-1).flatten()

    em = sum([1 if p == l else 0 for p, l in zip(squad_labels, squad_preds)]) / len(squad_labels)
    f1 = f1_score(squad_labels, squad_preds, average='macro')

    return { 'exact_match': em, 'f1': f1 }

# Build Trainer
def build_trainer(tokenizer=tokenizer, model=model):
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1, # !very important to send data to cpu
        warmup_steps=1,
        num_train_epochs=4,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=output_dir,
        group_by_length=False,
        dataloader_drop_last=False,
        eval_strategy='steps'
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        eval_dataset=train_dataset,
        args=train_args,
        max_seq_length=cutoff_len,
        compute_metrics=compute_metrics
    )
    return trainer

trainer = build_trainer()
# Eval model
gc.collect()
gc.collect()
# try:
results = trainer.evaluate()
# except Exception as e:
#     torch.cuda.empty_cache()  # Clear the GPU cache before retrying
#     gc.collect()
#     gc.collect()
#     try:
#         results = trainer.evaluate()  # Retry evaluation
#     except Exception as e:
#         raise e  # If it fails again, raise the error

print(results)
