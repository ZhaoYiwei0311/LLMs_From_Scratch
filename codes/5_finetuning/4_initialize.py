import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_download import download_and_load_gpt2, load_gpt2_from_directory
from utils import generate, load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from gptModel import GPTModel
from config import GPT_CONFIG_124M, model_configs
from utils import generate_text_simple

import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_size = "124M"
settings, params = load_gpt2_from_directory(model_dir="gpt2/124M")

model_name = "gpt2-small (124M)" 
NEW_CONFIG = GPT_CONFIG_124M.copy() 
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

# gpt = GPTModel(NEW_CONFIG) 
# gpt.eval()

gpt = GPTModel(NEW_CONFIG)

load_weights_into_gpt(gpt, params) 
gpt.eval()

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    gpt, 
    idx=text_to_token_ids(text_1, tokenizer), 
    max_new_tokens=15, 
    context_size=NEW_CONFIG["context_length"]
    )

text_2 = ( 
    "Is the following text 'spam'? Answer with 'yes' or 'no':" 
    " 'You are a winner you have been specially" 
    " selected to receive $1000 cash or a $2000 award.'" 
) 
token_ids = generate_text_simple( 
    model=gpt, 
    idx=text_to_token_ids(text_2, tokenizer), 
    max_new_tokens=23, 
    context_size=NEW_CONFIG["context_length"] 
) 
print(token_ids_to_text(token_ids, tokenizer))