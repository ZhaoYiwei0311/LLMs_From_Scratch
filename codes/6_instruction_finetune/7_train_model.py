import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_download import load_gpt2_from_directory
from gptModel import GPTModel, train_model_simple
from utils import format_input, load_weights_into_gpt, calc_loss_loader
from instructionData import InstructionDataset, custom_collate_fn
from torch.utils.data import DataLoader
import json
from functools import partial

import torch
import tiktoken

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
model.eval()
model.to(device)


with torch.no_grad(): 
    train_loss = calc_loss_loader( 
        train_loader, model, device, num_batches=5 
    ) 
    val_loss = calc_loss_loader( 
        val_loader, model, device, num_batches=5 
) 
print("Training loss:", train_loss) 
print("Validation loss:", val_loss)

import time 
start_time = time.time() 
torch.manual_seed(123) 
optimizer = torch.optim.AdamW( 
    model.parameters(), lr=0.00005, weight_decay=0.1 
) 
num_epochs = 2 
train_losses, val_losses, tokens_seen = train_model_simple( 
    model, train_loader, val_loader, optimizer, device, 
    num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
    start_context=format_input(val_data[0]), tokenizer=tokenizer 
) 
end_time = time.time() 
execution_time_minutes = (end_time - start_time) / 60 
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

torch.save({ 
    "model_state_dict": model.state_dict(), 
    "optimizer_state_dict": optimizer.state_dict(), 
    }, 
    "gpt_model_instruction_finetune.pth" 
)