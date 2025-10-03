import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from utils import format_input

data = json.load(open("data/instruction-data.json"))


model_input = format_input(data[50]) 
desired_response = f"\n\n### Response:\n{data[50]['output']}" 
print(model_input + desired_response)