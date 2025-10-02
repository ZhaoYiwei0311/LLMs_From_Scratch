import tiktoken
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptModel import DummyGRTModel

tokenizer = tiktoken.get_encoding("gpt2") 
batch = [] 
txt1 = "Every effort moves you" 
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1))) 
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0) 
print(batch) 

torch.manual_seed(123)
model = DummyGRTModel()
logits = model(batch)
print("Output shape:", logits.shape) 
print(logits) 