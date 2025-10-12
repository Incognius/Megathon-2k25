import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

# Import the device setup from our existing file
from .device import DEVICE

def train_epoch(model, dataloader, optimizer, criterion):
    """Performs one full training pass over the dataset."""
    model.train()
    total_loss = 0
    
    for batch, labels in tqdm(dataloader, desc="Training"):
        # Move data to the selected device (GPU/CPU)
        numeric_data = batch['numeric'].to(DEVICE)
        categorical_data = batch['categorical'].to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model({'numeric': numeric_data, 'categorical': categorical_data})
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion):
    """Performs one full evaluation pass."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in tqdm(dataloader, desc="Evaluating"):
            numeric_data = batch['numeric'].to(DEVICE)
            categorical_data = batch['categorical'].to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            outputs = model({'numeric': numeric_data, 'categorical': categorical_data})
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions (apply sigmoid and threshold at 0.5)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc
    }
    
    return metrics
