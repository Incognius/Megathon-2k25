import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import argparse
import os

def main(args):
    """
    Trains and evaluates an XGBoost model on the preprocessed data using the correct
    Scikit-Learn API for early stopping.
    """
    print("--- Starting XGBoost Model Training ---")

    # --- 1. Load Preprocessed Data ---
    print("Loading data from data/processed/...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    # --- 2. Initialize XGBoost Model ---
    print("Initializing XGBoost model with GPU support...")
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        eval_metric='logloss',
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=args.n_estimators,
        random_state=42,
        use_label_encoder=False # This is deprecated and should be False
    )

    # --- 3. Train the Model with Correct Early Stopping Syntax ---
    print("Training the model with early stopping...")
    
    # The validation set must be passed within a list to the 'eval_set' parameter.
    eval_set = [(X_val, y_val)]
    
    # The early_stopping_rounds parameter is passed to .fit() in modern versions.
    # If it's not working, it indicates an older version is still being used.
    # Let's add a check for robustness.
    try:
        # This is the modern, preferred way for versions >= 1.3.0
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  early_stopping_rounds=50,
                  verbose=True)
    except TypeError:
        # This is a fallback for older versions of XGBoost.
        print("\n[Warning] `early_stopping_rounds` not supported directly in .fit(). Falling back to older syntax.")
        print("Consider upgrading XGBoost with: pip install --upgrade xgboost\n")
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  # The parameter was named 'early_stopping_rounds' in the call, but it's a fit param
                  # In very old versions, this wasn't possible at all in the sklearn API.
                  # The most compatible way is to simply remove it if the modern API fails.
                  verbose=True)


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
    print(f"\nXGBoost model saved to {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost Churn Prediction Model.")
    parser.add_argument("--model-save-path", type=str, default="artifacts/models/best_churn_model_xgb.joblib")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    
    args = parser.parse_args()
    main(args)