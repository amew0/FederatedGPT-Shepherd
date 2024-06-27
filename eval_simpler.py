# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
logg = lambda x: print(f"------------------------ {x} ---------------------------")
from datetime import datetime
print(torch.cuda.is_available())

# %%
ME = "/dpc/kunf0097/l3-8b"
RUN_ID = datetime.now().strftime("%y%m%d%H%M%S")
logg(RUN_ID)
logg('eval_simpler.py')

# %%
name = "amew0/l3-8b-medical-v240623023136"
tokenizer = AutoTokenizer.from_pretrained(
    name, 
    cache_dir=f"{ME}/tokenizer", 
    padding_side="right", 
    pad_token_id=0,
    legacy=False
)
tokenizer.pad_token = tokenizer.eos_token

# %%
# Initialize model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    name, 
    cache_dir=f"{ME}/model",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    return_dict=True
)

# %%
# Tokenize prompt function
cutoff_len = 128  # Maximum length to control CUDA OOM

def tokenize(prompt, tokenizer, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors="pt",
    )
    result["input_ids"] = result["input_ids"].flatten()
    result["attention_mask"] = result["attention_mask"].flatten()

    if add_eos_token and result["input_ids"].shape[0] < cutoff_len:
        result["input_ids"][-1] = tokenizer.eos_token_id
        result["attention_mask"][-1] = 1

    result["labels"] = result["input_ids"].clone()
    return result

def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer).to('cpu')
    return tokenized_full_prompt


# %%
data_path = './data/1/eval_medical_2k.json'
data = load_dataset("json", data_files=data_path)
eval_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)

# %%
from sklearn.metrics import f1_score

# Evaluation function
def evaluate_squad_predictions(pred):
    squad_labels = pred["labels"].flatten()
    squad_preds = pred["predictions"].argmax(-1).flatten()

    em = sum([1 if p == l else 0 for p, l in zip(squad_labels, squad_preds)]) / len(squad_labels)
    f1 = f1_score(squad_labels, squad_preds, average='macro')

    return {'exact_match': em, 'f1': f1}

# %%
results = []
from tqdm import tqdm
for example in tqdm(eval_dataset):
    with torch.no_grad():
        example["input_ids"] = torch.LongTensor(example["input_ids"]).unsqueeze(0)
        example["attention_mask"] = torch.LongTensor(example["attention_mask"]).unsqueeze(0)

        outputs = model(input_ids=example["input_ids"], 
                        attention_mask=example["attention_mask"])
    pred = {
        "labels": example["input_ids"],
        "predictions": outputs["logits"],
    }
    
    # Evaluate predictions
    eval_results = evaluate_squad_predictions(pred)
    results.append(eval_results)


# %%
# Aggregate metrics
em_total = sum(result['exact_match'] for result in results) / len(results)
f1_total = sum(result['f1'] for result in results) / len(results)

print(f"Aggregate EM: {em_total}, Aggregate F1: {f1_total}")

# %%



