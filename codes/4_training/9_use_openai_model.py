import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_download import download_and_load_gpt2
from utils import generate, load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from gptModel import GPTModel
from config import GPT_CONFIG_124M, model_configs

import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_size = "124M"
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

model_name = "gpt2-small (124M)" 
NEW_CONFIG = GPT_CONFIG_124M.copy() 
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

# gpt = GPTModel(NEW_CONFIG) 
# gpt.eval()

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params) 
gpt.to(device)

torch.manual_seed(123) 
token_ids = generate( 
    model=gpt, 
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device), 
    max_new_tokens=25, 
    context_size=NEW_CONFIG["context_length"], 
    top_k=50, 
    temperature=1.5 
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer)) 