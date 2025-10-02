import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_weights_into_gpt
from gpt_download import load_gpt2_from_directory
from config import GPT_CONFIG_124M, model_configs
from spamData import SpamDataset, classify_review
from gptModel import GPTModel

import torch
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_size = "124M"
model_name = "gpt2-small (124M)" 
settings, params = load_gpt2_from_directory(model_dir="gpt2/124M")
NEW_CONFIG = GPT_CONFIG_124M.copy() 
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

model = GPTModel(NEW_CONFIG)
load_weights_into_gpt(model, params)
model.load_state_dict(torch.load("model/review_classifier.pth"))
model = model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")

text_1 = ( 
    "You are a winner you have been specially" 
    " selected to receive $1000 cash or a $2000 award." 
) 

train_dataset = SpamDataset( 
    csv_file="data/spam/train.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

print(classify_review( 
    text_1, model, tokenizer, device, max_length=train_dataset.max_length 
))