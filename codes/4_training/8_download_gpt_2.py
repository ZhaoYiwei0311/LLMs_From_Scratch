import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2( 
    model_size="124M", models_dir="gpt2" 
)

print("Settings:", settings) 
print("Parameter dictionary keys:", params.keys())
print(params["wte"]) 
print("Token embedding weight tensor dimensions:", params["wte"].shape) 