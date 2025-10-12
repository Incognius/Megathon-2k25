import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import optuna
import argparse
import os

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.churnxai.dataset import ChurnDataset
from src.churnxai.model import ChurnPredictor
from src.churnxai.train import train_epoch, evaluate_model
from src.churnxai.device import DEVICE

def objective(trial: optuna.Trial):
    """
    The objective function for Optuna to optimize.
    A "trial" represents one full training and evaluation run with one set of hyperparameters.
    """
    # --- 1. Suggest Hyperparameters ---
    # We define the search space for Optuna here.
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f"n_units_l{i}", 50, 200))

    # --- 2. Load Data ---
    # This is done inside the objective to ensure data is fresh for each trial.
    preprocessor = joblib.load("artifacts/models/preprocessor.joblib")
    train_dataset = ChurnDataset("data/processed/X_train.csv", "data/processed/y_train.csv", preprocessor)
    val_dataset = ChurnDataset("data/processed/X_val.csv", "data/processed/y_val.csv", preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=2)
    
    # --- 3. Build Model and Optimizer ---
    model = ChurnPredictor(
        preprocessor, 
        hidden_layers=hidden_layers, 
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Train and Evaluate ---
    # We'll train for a fixed number of epochs for each trial.
    epochs = 5 # Using fewer epochs for tuning to speed it up.
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        
    # Evaluate on the validation set
    metrics = evaluate_model(model, val_loader, criterion)
    
    # --- 5. Return the metric to optimize ---
    # We want to maximize the F1-score.
    return metrics['f1_score']

def main(args):
    print("--- Starting Hyperparameter Tuning with Optuna ---")
    
    # Create a study. "direction='maximize'" means Optuna will try to make the F1-score as high as possible.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- Tuning Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (F1-Score): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save the study object for later analysis
    joblib.dump(study, "artifacts/models/optuna_study.joblib")
    print("\nStudy saved to artifacts/models/optuna_study.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials to run.")
    args = parser.parse_args()
    main(args)