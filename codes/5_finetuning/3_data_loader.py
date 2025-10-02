import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spamData import SpamDataset
import tiktoken
import torch
from torch.utils.data import DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset( 
    csv_file="data/spam/train.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

val_dataset = SpamDataset( 
    csv_file="data/spam/validation.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

test_dataset = SpamDataset( 
    csv_file="data/spam/test.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataloader = DataLoader( 
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=num_workers 
)

val_dataloader = DataLoader( 
    val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers
)

test_dataloader = DataLoader( 
    test_dataset, 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers
)

for input_batch, target_batch in train_dataloader: 
    pass 
print("Input batch dimensions:", input_batch.shape) 
print("Label batch dimensions", target_batch.shape) 

print(f"{len(train_dataloader)} training batches") 
print(f"{len(val_dataloader)} validation batches") 
print(f"{len(test_dataloader)} test batches")