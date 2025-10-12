import pandas as pd
import argparse
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.churnxai.preprocess import ChurnDataPreprocessor

def run_prep(csv_path: str, target_col: str, n_clusters: int):
    """ Main function to run the full data preparation pipeline with a 75/10/15 split. """
    print("--- Starting Data Preparation Pipeline ---")
    raw_df = pd.read_csv(csv_path)
    
    print("\n--- Splitting Data (75% Train, 10% Val, 15% Test) ---")
    train_val_df, test_df = train_test_split(
        raw_df, test_size=0.15, random_state=42, stratify=raw_df[target_col]
    )
    val_size_relative = 0.10 / 0.85
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_relative, random_state=42, stratify=train_val_df[target_col]
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize and fit the now-unified preprocessor
    preprocessor = ChurnDataPreprocessor(n_clusters=n_clusters)
    preprocessor.fit(train_df, target_col=target_col)

    # Transform all data splits using the single, powerful transform method
    X_train = preprocessor.transform(train_df)
    y_train = train_df[target_col]
    
    X_val = preprocessor.transform(val_df)
    y_val = val_df[target_col]
    
    X_test = preprocessor.transform(test_df)
    y_test = test_df[target_col]
    
    print("\n--- Applying SMOTE to Training Data ONLY ---")
    print(f"Train class distribution before SMOTE: {Counter(y_train)}")
    smote = SMOTE(random_state=42) # Assuming you upgraded imbalanced-learn
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Train class distribution after SMOTE: {Counter(y_train_resampled)}")

    print("\n--- Saving Processed Data and Artifacts ---")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train_resampled, columns=X_train.columns).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.Series(y_train_resampled, name=target_col).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    
    X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    pd.Series(y_val, name=target_col).to_csv(os.path.join(output_dir, "y_val.csv"), index=False)

    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.Series(y_test, name=target_col).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Processed data saved to {output_dir}")
    
    preprocessor.save("artifacts/models/preprocessor.joblib")
    print("Unified preprocessor (including k-means model) saved.")

    print("\n--- Data Preparation Complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full data preparation pipeline for churn prediction.")
    parser.add_argument("--csv", type=str, default="data/raw/autoinsurance_churn.csv", help="Path to the input CSV file.")
    parser.add_argument("--target-col", type=str, default="Churn", help="Name of the target column.")
    parser.add_argument("--clusters", type=int, default=12, help="Number of geo clusters to create.")
    
    args = parser.parse_args()
    run_prep(csv_path=args.csv, target_col=args.target_col, n_clusters=args.clusters)