import pandas as pd
import xgboost as xgb
import joblib
import os

def main():
    """
    Trains the final champion model (XGBoost) using the best hyperparameters 
    found during the Optuna tuning study.
    """
    print("--- Creating Final Champion Model (XGBoost) ---")

    # --- 1. Load Preprocessed Data ---
    print("Loading all training data from data/processed/...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    # --- 2. Define Best Hyperparameters from Optuna ---
    # These are the winning parameters from your tuning run.
    best_params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'n_estimators': 1500, # Using a high number as we aren't using early stopping here
        
        # --- Winning parameters from your Optuna study ---
        'learning_rate': 0.01429066093843199,
        'max_depth': 6,
        'subsample': 0.7314363787361877,
        'colsample_bytree': 0.7868152748074655,
        'gamma': 1.095948644339276e-06,
        'min_child_weight': 8,
        # --- End of winning parameters ---

        'random_state': 42
    }
    print("Using best hyperparameters found from tuning:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # --- 3. Initialize and Train the Final Model ---
    model = xgb.XGBClassifier(**best_params)
    
    print("\nTraining final model on the full training dataset...")
    # We train on the full training set. No validation set is needed here.
    model.fit(X_train, y_train, verbose=True)

    # --- 4. Save the Final Model ---
    model_save_path = "artifacts/models/champion_model_xgb.joblib"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"\nFinal champion XGBoost model saved to {model_save_path}")

if __name__ == "__main__":
    main()