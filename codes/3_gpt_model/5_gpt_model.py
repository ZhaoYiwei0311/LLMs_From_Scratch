import torch
import tiktoken
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import GPTModel
from config import GPT_CONFIG_124M

torch.manual_seed(123) 
tokenizer = tiktoken.get_encoding("gpt2") 
batch = [] 
txt1 = "Every effort moves you" 
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1))) 
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0) 
model = GPTModel(GPT_CONFIG_124M) 
out = model(batch) 
print("Input batch:\n", batch) 
print("\nOutput shape:", out.shape) 
print(out)

total_params = sum(p.numel() for p in model.parameters()) 
print(f"Total number of parameters: {total_params:,}")
total_params_gpt2 = ( 
    total_params - sum(p.numel() 
    for p in model.out_head.parameters()) 
) 
print(f"Number of trainable parameters " 
      f"considering weight tying: {total_params_gpt2:,}" 
) 