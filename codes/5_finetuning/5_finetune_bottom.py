import sys
import os

from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spamData import SpamDataset, calc_loss_loader, calc_accuracy_loader
from gpt_download import download_and_load_gpt2, load_gpt2_from_directory
from utils import generate, load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from gptModel import GPTModel
from config import GPT_CONFIG_124M, model_configs
from utils import generate_text_simple

import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_size = "124M"
settings, params = load_gpt2_from_directory(model_dir="gpt2/124M")

model_name = "gpt2-small (124M)" 
NEW_CONFIG = GPT_CONFIG_124M.copy() 
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)

load_weights_into_gpt(gpt, params)
for param in gpt.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
gpt.out_head = torch.nn.Linear(
    in_features=NEW_CONFIG["emb_dim"],
    out_features=num_classes
)

for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in gpt.final_norm.parameters():
    param.requires_grad = True

gpt = gpt.to(device)  # Move model to device after all modifications

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0).to(device)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = gpt(inputs) 
print("Outputs:\n", outputs) 
print("Outputs dimensions:", outputs.shape)
print("Last output token:", outputs[:, -1, :])
probas = torch.softmax(outputs[:, -1, :], dim=-1) 
label = torch.argmax(probas) 
print("Class label:", label.item())

train_dataset = SpamDataset( 
    csv_file="data/spam/train.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

val_dataset = SpamDataset( 
    csv_file="data/spam/validation.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

test_dataset = SpamDataset( 
    csv_file="data/spam/test.csv", 
    max_length=None, 
    tokenizer=tokenizer 
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataloader = DataLoader( 
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=num_workers 
)

val_dataloader = DataLoader( 
    val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers
)

test_dataloader = DataLoader( 
    test_dataset, 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers
)

train_accuracy = calc_accuracy_loader( 
    train_dataloader, gpt, device, num_batches=10 
) 
val_accuracy = calc_accuracy_loader( 
    val_dataloader, gpt, device, num_batches=10 
) 
test_accuracy = calc_accuracy_loader( 
    test_dataloader, gpt, device, num_batches=10 
) 
print(f"Training accuracy: {train_accuracy*100:.2f}%") 
print(f"Validation accuracy: {val_accuracy*100:.2f}%") 
print(f"Test accuracy: {test_accuracy*100:.2f}%")

with torch.no_grad(): 
    train_loss = calc_loss_loader( 
        train_dataloader, gpt, device, num_batches=5 
    ) 
    val_loss = calc_loss_loader(val_dataloader, gpt, device, num_batches=5) 
    test_loss = calc_loss_loader(test_dataloader, gpt, device, num_batches=5) 
print(f"Training loss: {train_loss:.3f}") 
print(f"Validation loss: {val_loss:.3f}") 
print(f"Test loss: {test_loss:.3f}")