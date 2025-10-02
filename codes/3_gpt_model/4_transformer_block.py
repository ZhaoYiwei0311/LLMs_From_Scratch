import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import TransformerBlock
from config import GPT_CONFIG_124M


torch.manual_seed(123) 
x = torch.rand(2, 4, 768) 
block = TransformerBlock(GPT_CONFIG_124M) 
output = block(x) 
print("Input shape:", x.shape) 
print("Output shape:", output.shape)