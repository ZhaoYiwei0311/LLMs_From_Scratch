import sys
import os
import tiktoken
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptModel import generate_text_simple, GPTModel
from utils import text_to_token_ids, token_ids_to_text
from config import GPT_CONFIG_124M


model = GPTModel(GPT_CONFIG_124M)

start_context = "Every effort moves you" 
tokenizer = tiktoken.get_encoding("gpt2") 
token_ids = generate_text_simple( 
    model=model, 
    idx=text_to_token_ids(start_context, tokenizer), 
    max_new_tokens=10, 
    context_size=GPT_CONFIG_124M["context_length"] 
) 
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))