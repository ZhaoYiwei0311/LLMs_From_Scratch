import torch
import sys
import os
import tiktoken
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import GPTModel
from utils import text_to_token_ids, token_ids_to_text
from config import GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")
inputs = torch.tensor(
    [[16833, 3626, 6100],   # ["every effort moves", 
    [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor(
    [[3626, 6100, 345  ],  # [" effort moves you", 
    [1107, 588, 11311]])  #  " really like chocolate"]  

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True) 
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}") 
print(f"Outputs batch 1:" 
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0 
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]] 
print("Text 1:", target_probas_1) 
text_idx = 1 
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]] 
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2))) 
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1 
print(neg_avg_log_probas) 

print("Logits shape:", logits.shape) 
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1) 
targets_flat = targets.flatten() 
print("Flattened logits:", logits_flat.shape) 
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat) 
print(loss)