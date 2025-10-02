import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate, text_to_token_ids, token_ids_to_text
from gptModel import GPTModel
from config import GPT_CONFIG_124M
import tiktoken

next_token_logits = torch.tensor( 
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79] 
) 

top_k = 3 
top_logits, top_pos = torch.topk(next_token_logits, top_k) 
print("Top logits:", top_logits) 
print("Top positions:", top_pos)

new_logits = torch.where( 
    condition=next_token_logits < top_logits[-1], 
    input=torch.tensor(float('-inf')), 
    other=next_token_logits 
) 
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0) 
print(topk_probas)

model = GPTModel(GPT_CONFIG_124M)
state_dict = torch.load("gpt_model.pth", weights_only=True)
model.load_state_dict(state_dict)
model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2") 

torch.manual_seed(123) 
token_ids = generate( 
    model=model, 
    idx=text_to_token_ids("Every effort moves you", tokenizer), 
    max_new_tokens=15, 
    context_size=GPT_CONFIG_124M["context_length"], 
    top_k=25, 
    temperature=5
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))