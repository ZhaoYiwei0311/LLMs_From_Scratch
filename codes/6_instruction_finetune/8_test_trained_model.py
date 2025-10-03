import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptModel import GPTModel
from utils import format_input, text_to_token_ids, token_ids_to_text, generate, load_weights_into_gpt
from gpt_download import load_gpt2_from_directory
from functools import partial
from instructionData import InstructionDataset, custom_collate_fn
import tiktoken
import json
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

customized_collate_fn = partial( 
    custom_collate_fn, 
    device=device, 
    allowed_max_length=1024 
)
torch.manual_seed(123)

data = json.load(open("data/instruction-data.json"))

train_portion = int(len(data) * 0.85) 
test_portion = int(len(data) * 0.1) 
val_portion = len(data) - train_portion - test_portion 

train_data = data[:train_portion] 
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer) 
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

BASE_CONFIG = { 
    "vocab_size": 50257,    # vocab size
    "context_length": 1024, # context length
    "drop_rate": 0.0,       # dropout 
    "qkv_bias": True        # bias
} 

model_configs = { 
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
} 

CHOOSE_MODEL = "gpt2-medium (355M)" 
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")") 
settings, params = load_gpt2_from_directory(model_dir="gpt2/355M")
model = GPTModel(BASE_CONFIG) 
load_weights_into_gpt(model, params)
model.load_state_dict(torch.load("model/gpt_model_instruction_finetune.pth")["model_state_dict"])
model.eval()
model.to(device)


torch.manual_seed(123) 
for entry in test_data[:3]: 
    input_text = format_input(entry) 
    token_ids = generate( 
        model=model, 
        idx=text_to_token_ids(input_text, tokenizer).to(device), 
        max_new_tokens=256, 
        context_size=BASE_CONFIG["context_length"], 
        eos_id=50256 
    ) 
    generated_text = token_ids_to_text(token_ids, tokenizer) 
    print('generated_text: ', generated_text)
    response_text = ( 
        generated_text[len(input_text):] 
        .replace("### Response:", "") 
        .strip() 
    ) 
    print(input_text) 
    print(f"\nCorrect response:\n>> {entry['output']}") 
    print(f"\nModel response:\n>> {response_text.strip()}") 
    print("-------------------------------------") 
