import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import argparse
import os

def main(args):
    """
    Trains and evaluates a LightGBM model on the preprocessed data.
    """
    print("--- Starting LightGBM Model Training ---")

    # --- 1. Load Preprocessed Data ---
    print("Loading data from data/processed/...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    # --- 2. Initialize LightGBM Model ---
    # Using the robust parameters you had before, with GPU acceleration
    print("Initializing LightGBM model with GPU support...")
    model = lgb.LGBMClassifier(
        device='gpu',
        boosting_type='gbdt',
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=args.n_estimators,
        random_state=42
    )

    # --- 3. Train the Model with Early Stopping ---
    print("Training the model with early stopping...")
    
    # LightGBM uses a 'callbacks' list for early stopping in its scikit-learn API
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              callbacks=callbacks)

    # --- 4. Evaluate on the Validation Set ---
    print("\n--- Evaluating on Validation Set ---")
    y_pred = model.predict(X_val)
    
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred)
    }
    
    print("Validation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # --- 5. Save the Model ---
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    joblib.dump(model, args.model_save_path)
    print(f"\nLightGBM model saved to {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LightGBM Churn Prediction Model.")
    parser.add_argument("--model-save-path", type=str, default="artifacts/models/best_churn_model_lgbm.joblib")
    parser.add_argument("--n-estimators", type=int, default=1000) # Increased for early stopping
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    
    args = parser.parse_args()
    main(args)