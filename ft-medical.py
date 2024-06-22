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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(os.getenv('PYTORCH_CUDA_ALLOC_CONF'))

start = time()

# Load environment variables
load_dotenv()

# Hugging Face token
HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")

# Login to Hugging Face Hub
huggingface_hub.login(token=HF_TOKEN_WRITE)


torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

# Set model and tokenizer paths
ME = "/dpc/kunf0097/l3-8b"
RUN_ID = datetime.now().strftime("%y%m%d%H%M%S")

# Initialize tokenizer
name = "meta-llama/Meta-Llama-3-8B"

# Initialize model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    name, 
    cache_dir=f"{ME}/l3-8b/model",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    return_dict=True
)

tokenizer = AutoTokenizer.from_pretrained(
    name, 
    cache_dir=f"{ME}/l3-8b/tokenizer", 
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
    tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer).to('cuda:1')
    return tokenized_full_prompt

# Load and process dataset
data_path = './data/1/medical.json'
output_dir = f'{ME}/output/output_{RUN_ID}'
data = load_dataset("json", data_files=data_path)
train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)

# Build Trainer
def build_trainer(tokenizer=tokenizer, model=model):
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1, # !very important to send data to cpu
        warmup_steps=1,
        num_train_epochs=1,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=output_dir,
        group_by_length=False,
        dataloader_drop_last=False
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        args=train_args,
        max_seq_length=cutoff_len,
    )
    return trainer

trainer = build_trainer()
# Train model
gc.collect()
gc.collect()
trainer.train()

# Save model and tokenizer
model_save_path = f"{ME}/model/l3-8b-medical-v{RUN_ID}"
trainer.model.save_pretrained(model_save_path)

# saving to load later from https://www.youtube.com/watch?v=Pb_RGAl75VE&ab_channel=DataCamp
model = AutoModelForCausalLM.from_pretrained(
    name, 
    cache_dir=f"{ME}/l3-8b/model",
    torch_dtype=torch.float16,
    device_map={"": 0},
    return_dict=True
)
model = PeftModel.from_pretrained(model,model_save_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(
    name, 
    cache_dir=f"{ME}/l3-8b/tokenizer", 
    padding_side="right", 
    pad_token_id=(0),
    legacy=False
)
tokenizer.pad_token = tokenizer.eos_token

# Push to Hugging Face Hub
tokenizer.push_to_hub(f"l3-8b-medical-v{RUN_ID}", token=HF_TOKEN_WRITE)
model.push_to_hub(f"l3-8b-medical-v{RUN_ID}", token=HF_TOKEN_WRITE)

# Log elapsed time
end = time()
print(f"Elapsed time: {end - start}")
print(f"Run ID: {RUN_ID}")
