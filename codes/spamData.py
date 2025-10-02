import torch 
from torch.utils.data import Dataset 
import pandas as pd

class SpamDataset(Dataset): 
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256): 
        self.data = pd.read_csv(csv_file) 
        
        self.encoded_texts = [ 
            tokenizer.encode(text) for text in self.data["Text"] 
        ] 
        if max_length is None: 
            self.max_length = self._longest_encoded_length() 
        else: 
            self.max_length = max_length 
            
            self.encoded_texts = [ 
                encoded_text[:self.max_length] 
                for encoded_text in self.encoded_texts 
            ] 
            
        self.encoded_texts = [ 
            encoded_text + [pad_token_id] * 
            (self.max_length - len(encoded_text)) 
            for encoded_text in self.encoded_texts 
        ] 
        
    def __getitem__(self, index): 
        encoded = self.encoded_texts[index] 
        label = self.data.iloc[index]["Label"] 
        return ( 
            torch.tensor(encoded, dtype=torch.long), 
            torch.tensor(label, dtype=torch.long) 
        ) 
        
    def __len__(self): 
        return len(self.data) 
    
    def _longest_encoded_length(self): 
        max_length = 0 
        for encoded_text in self.encoded_texts: 
            encoded_length = len(encoded_text) 
            if encoded_length > max_length: 
                max_length = encoded_length 
        return max_length 

def calc_accuracy_loader(data_loader, model, device, num_batches=None): 
    # model.eval() 
    correct_predictions, num_examples = 0, 0 
    if num_batches is None: 
        num_batches = len(data_loader) 
    else: 
        num_batches = min(num_batches, len(data_loader)) 
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            input_batch = input_batch.to(device) 
            target_batch = target_batch.to(device) 
            with torch.no_grad(): 
                logits = model(input_batch)[:, -1, :] 
            predicted_labels = torch.argmax(logits, dim=-1) 
            num_examples += predicted_labels.shape[0] 
            correct_predictions += ( 
                (predicted_labels == target_batch).sum().item() 
            ) 
        else: 
            break 
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device): 
    input_batch = input_batch.to(device) 
    target_batch = target_batch.to(device) 
    logits = model(input_batch)[:, -1, :] 
    loss = torch.nn.functional.cross_entropy(logits, target_batch) 
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None): 
    total_loss = 0. 
    if len(data_loader) == 0: 
        return float("nan") 
    elif num_batches is None: 
        num_batches = len(data_loader) 
    else: 
        num_batches = min(num_batches, len(data_loader)) 
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            loss = calc_loss_batch( 
                input_batch, target_batch, model, device 
            ) 
            total_loss += loss.item() 
        else: 
            break 
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter): 
    model.eval() 
    with torch.no_grad(): 
        train_loss = calc_loss_loader( 
            train_loader, model, device, num_batches=eval_iter 
        ) 
        val_loss = calc_loss_loader( 
            val_loader, model, device, num_batches=eval_iter 
        ) 
    model.train() 
    return train_loss, val_loss

def train_classifier_simple( 
        model, train_loader, val_loader, optimizer, device, 
        num_epochs, eval_freq, eval_iter):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] 
    examples_seen, global_step = 0, -1
    
    for epoch in range(num_epochs): 
        model.train()
        
    for input_batch, target_batch in train_loader: 
        optimizer.zero_grad() 
        loss = calc_loss_batch( 
            input_batch, target_batch, model, device 
        ) 
        loss.backward() 
        optimizer.step() 
        examples_seen += input_batch.shape[0] 
        global_step += 1
        
        if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model( 
                    model, train_loader, val_loader, device, eval_iter) 
                train_losses.append(train_loss) 
                val_losses.append(val_loss) 
                print(f"Ep {epoch+1} (Step {global_step:06d}): " 
                      f"Train loss {train_loss:.3f}, " 
                      f"Val loss {val_loss:.3f}" 
                ) 
                
        train_accuracy = calc_accuracy_loader( 
            train_loader, model, device, num_batches=eval_iter 
        ) 
        val_accuracy = calc_accuracy_loader( 
            val_loader, model, device, num_batches=eval_iter 
        ) 
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="") 
        print(f"Validation accuracy: {val_accuracy*100:.2f}%") 
        train_accs.append(train_accuracy) 
        val_accs.append(val_accuracy) 
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def classify_review( 
        text, model, tokenizer, device, max_length=None, 
        pad_token_id=50256): 
    model.eval() 
    
    input_ids = tokenizer.encode(text) 
    supported_context_length = model.pos_emb.weight.shape[1] 
    input_ids = input_ids[:min( 
        max_length, supported_context_length 
    )]
    
    input_ids += [pad_token_id] * (max_length - len(input_ids)) 
    input_tensor = torch.tensor( 
        input_ids, device=device 
    ).unsqueeze(0) 
    
    with torch.no_grad(): 
        logits = model(input_tensor)[:, -1, :] 
    predicted_label = torch.argmax(logits, dim=-1).item() 
    return "spam" if predicted_label == 1 else "not spam"