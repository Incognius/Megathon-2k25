import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import optuna
import joblib
import argparse
import os

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val):
    """
    The objective function for Optuna to optimize for XGBoost.
    """
    # --- 1. Suggest Hyperparameters for XGBoost ---
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'n_estimators': 1000, # High n_estimators with early stopping is best practice
        
        # Parameters to be tuned by Optuna
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    # --- 2. Build and Train the Model ---
    model = xgb.XGBClassifier(**params, random_state=42)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              
              verbose=False) # Keep the output clean during tuning

    # --- 3. Evaluate and Return Metric ---
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    
    return f1


def main(args):
    """
    Main function to run the Optuna tuning study for XGBoost.
    """
    print("--- Starting XGBoost Hyperparameter Tuning with Optuna ---")
    
    # --- Load Data Once ---
    print("Loading data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()
    
    # --- Run Study ---
    # We pass the data to the objective function using a lambda to avoid reloading it every time
    study_objective = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(study_objective, n_trials=args.n_trials)

    print("\n--- Tuning Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (F1-Score): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save the study object for later analysis
    output_path = "artifacts/models/optuna_study_xgb.joblib"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(study, output_path)
    print(f"\nXGBoost study saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for XGBoost with Optuna.")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of optimization trials to run.")
    args = parser.parse_args()
    main(args)