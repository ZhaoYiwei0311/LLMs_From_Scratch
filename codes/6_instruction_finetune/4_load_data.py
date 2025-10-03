from torch.utils.data import DataLoader
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import partial
from instructionData import InstructionDataset, custom_collate_fn
import tiktoken
import json

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

print("Train loader:") 
for inputs, targets in train_loader: 
    print(inputs.shape, targets.shape)