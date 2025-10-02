import tiktoken
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataLoader import create_dataloader_v1
import torch
from config import GPT_CONFIG_124M
from gptModel import GPTModel
from utils import calc_loss_loader

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "data/the-verdict.txt"

with open(file_path, "r") as file:
    text_data = file.read()


total_characters = len(text_data) 
total_tokens = len(tokenizer.encode(text_data)) 

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1( 
    train_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    drop_last=True, 
    shuffle=True, 
    num_workers=0 
) 
val_loader = create_dataloader_v1( 
    val_data, 
    batch_size=2, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"], 
    drop_last=False, 
    shuffle=False, 
    num_workers=0 
)

print("Train loader:") 
for x, y in train_loader: 
    print(x.shape, y.shape) 
print("\nValidation loader:") 
for x, y in val_loader: 
    print(x.shape, y.shape)
    

model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 
with torch.no_grad(): 
    train_loss = calc_loss_loader(train_loader, model, device) 
    val_loss = calc_loss_loader(val_loader, model, device) 
print("Training loss:", train_loss) 
print("Validation loss:", val_loss)