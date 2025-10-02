import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_text_simple, text_to_token_ids, token_ids_to_text
from gptModel import GPTModel, train_model_simple
from config import GPT_CONFIG_124M
import tiktoken
from dataLoader import create_dataloader_v1

file_path = "data/the-verdict.txt"

with open(file_path, "r") as file:
    text_data = file.read()

tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data) 
total_tokens = len(tokenizer.encode(text_data)) 

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))

train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

torch.manual_seed(123) 
model = GPTModel(GPT_CONFIG_124M) 
model.to(device) 
optimizer = torch.optim.AdamW( 
     model.parameters(), 
    lr=0.0004, weight_decay=0.1 
) 

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

num_epochs = 10 
train_losses, val_losses, tokens_seen = train_model_simple( 
    model, train_loader, val_loader, optimizer, device, 
    num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
    start_context="Every effort moves you", tokenizer=tokenizer 
)

torch.save({ 
    "model_state_dict": model.state_dict(), 
    "optimizer_state_dict": optimizer.state_dict(), 
    }, 
    "gpt_model.pth" 
)
