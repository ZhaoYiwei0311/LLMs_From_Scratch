import torch
import sys
import os
import tiktoken
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptModel import GPTModel
from config import GPT_CONFIG_124M
from dataLoader import create_dataloader_v1
from utils import generate_and_print_sample, generate_text_simple, text_to_token_ids, token_ids_to_text

model = GPTModel(GPT_CONFIG_124M)
state_dict = torch.load("gpt_model.pth", weights_only=True)
model.load_state_dict(state_dict)
model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2") 
token_ids = generate_text_simple( 
    model=model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer), 
    max_new_tokens=25, 
    context_size=GPT_CONFIG_124M["context_length"] 
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))