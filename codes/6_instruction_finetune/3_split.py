import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from utils import format_input
from instructionData import custom_collate_draft_1, custom_collate_draft_2, custom_collate_fn

data = json.load(open("data/instruction-data.json"))

train_portion = int(len(data) * 0.85) 
test_portion = int(len(data) * 0.1) 
val_portion = len(data) - train_portion - test_portion 

train_data = data[:train_portion] 
test_data = data[train_portion:train_portion + test_portion] 
val_data = data[train_portion + test_portion:] 
print("Training set length:", len(train_data)) 
print("Validation set length:", len(val_data)) 
print("Test set length:", len(test_data))

inputs_1 = [0, 1, 2, 3, 4] 
inputs_2 = [5, 6] 
inputs_3 = [7, 8, 9] 
batch = ( 
    inputs_1, 
    inputs_2, 
    inputs_3 
) 
# print(custom_collate_draft_1(batch, device='cuda')) 

inputs, targets = custom_collate_fn(batch, device='cuda')
print(inputs)
print(targets)