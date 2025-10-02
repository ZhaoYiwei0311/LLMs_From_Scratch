import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import DummyLayerNorm

torch.manual_seed(123) 
batch_example = torch.randn(2, 5) 
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) 
out = layer(batch_example) 
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True) 
print("Mean:\n", mean) 
print("Variance:\n", var)

out_norm = (out - mean) / torch.sqrt(var) 
mean = out_norm.mean(dim=-1, keepdim=True) 
var = out_norm.var(dim=-1, keepdim=True) 
torch.set_printoptions(sci_mode=False) 

print("Normalized layer outputs:\n", out_norm) 
print("Mean:\n", mean) 
print("Variance:\n", var)

ln = DummyLayerNorm(emb_dim=5) 
out_ln = ln(batch_example) 
mean = out_ln.mean(dim=-1, keepdim=True) 
var = out_ln.var(dim=-1, unbiased=False, keepdim=True) 
print("Mean:\n", mean) 
print("Variance:\n", var)