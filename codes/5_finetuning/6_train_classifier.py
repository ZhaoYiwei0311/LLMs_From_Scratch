import time 
import torch
import sys
import os

from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spamData import SpamDataset, calc_accuracy_loader, train_classifier_simple
from gpt_download import load_gpt2_from_directory
from utils import load_weights_into_gpt
from gptModel import GPTModel
from config import GPT_CONFIG_124M, model_configs

import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

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
model = model.to(device)

start_time = time.time() 
torch.manual_seed(123) 
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) 
num_epochs = 5 

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

train_dataloader = DataLoader( 
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    drop_last=True, 
    num_workers=0 
)

val_dataloader = DataLoader( 
    val_dataset, 
    batch_size=8, 
    shuffle=False, 
    drop_last=True, 
    num_workers=0 
)

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple( 
        model, train_dataloader, val_dataloader, optimizer, device, 
        num_epochs=num_epochs, eval_freq=50, 
        eval_iter=5 
    ) 
end_time = time.time() 
execution_time_minutes = (end_time - start_time) / 60 
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import matplotlib.pyplot as plt 
def plot_values( 
        epochs_seen, examples_seen, train_values, val_values, 
        label="loss"): 
    fig, ax1 = plt.subplots(figsize=(5, 3)) 

    ax1.plot(epochs_seen, train_values, label=f"Training {label}") 
    ax1.plot( 
        epochs_seen, val_values, linestyle="-.", 
        label=f"Validation {label}" 
    ) 
    ax1.set_xlabel("Epochs") 
    ax1.set_ylabel(label.capitalize()) 
    ax1.legend() 
    ax2 = ax1.twiny() 
    ax2.plot(examples_seen, train_values, alpha=0) 
    ax2.set_xlabel("Examples seen") 
    fig.tight_layout() 
    plt.savefig(f"{label}-plot.pdf") 
    plt.show() 
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses)) 
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses)) 
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

train_accuracy = calc_accuracy_loader(train_dataloader, model, device) 
val_accuracy = calc_accuracy_loader(val_dataloader, model, device) 
# test_accuracy = calc_accuracy_loader(test_dataloader, model, device) 
print(f"Training accuracy: {train_accuracy*100:.2f}%") 
print(f"Validation accuracy: {val_accuracy*100:.2f}%") 
# print(f"Test accuracy: {test_accuracy*100:.2f}%")

torch.save(model.state_dict(), "review_classifier.pth") 