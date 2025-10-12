import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import os
import argparse

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from src.churnxai.dataset import ChurnDataset
from src.churnxai.model import ChurnPredictor
from src.churnxai.train import train_epoch, evaluate_model
from src.churnxai.device import DEVICE # Use our configured device (GPU/CPU)

def main(args):
    print("--- Starting PyTorch Model Training ---")
    
    # --- 1. Load Preprocessor and Datasets ---
    preprocessor = joblib.load(args.preprocessor_path)
    
    train_dataset = ChurnDataset("data/processed/X_train.csv", "data/processed/y_train.csv", preprocessor)
    val_dataset = ChurnDataset("data/processed/X_val.csv", "data/processed/y_val.csv", preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 2. Initialize Model, Optimizer, and Loss Function ---
    model = ChurnPredictor(preprocessor).to(DEVICE)
    print(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # BCEWithLogitsLoss is numerically more stable than using a Sigmoid layer + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # --- 3. Training Loop ---
    best_f1_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}")
        
        val_metrics = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch} Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
            
        # Save the best model based on validation F1-score
        if val_metrics['f1_score'] > best_f1_score:
            best_f1_score = val_metrics['f1_score']
            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            print(f"*** New best model saved with F1-score: {best_f1_score:.4f} ***")

    print("\n--- Training Complete ---")
    print(f"Best validation F1-score achieved: {best_f1_score:.4f}")
    print(f"Best model saved to {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PyTorch Churn Prediction Model.")
    parser.add_argument("--preprocessor-path", type=str, default="artifacts/models/preprocessor.joblib")
    parser.add_argument("--model-save-path", type=str, default="artifacts/models/best_churn_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)