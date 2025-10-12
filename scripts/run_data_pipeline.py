import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Add the src directory to the Python path to import our preprocessor
# This makes the script runnable from the project root (e.g., `python -m scripts.run_data_pipeline`)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from src.churnxai.preprocess import ChurnDataPreprocessor

def run_data_pipeline():
    """
    Loads the raw data, performs a stratified train-val-test split,
    fits the preprocessor on the training data, transforms all splits,
    and saves the processed data and the fitted preprocessor artifact.
    """
    print("--- Starting Data Processing Pipeline ---")

    # --- 1. Define Paths using pathlib for robustness ---
    raw_data_path = project_root / "data/raw/autoinsurance_churn.csv"
    processed_data_path = project_root / "data/processed"
    artifacts_path = project_root / "artifacts/models"
    
    # Create output directories if they don't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Raw Data ---
    try:
        print(f"Loading raw data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_data_path}. Please check the path.")
        return

    # --- 3. Stratified Train-Validation-Test Split (80/10/10) ---
    print("Performing stratified 80/10/10 train-validation-test split...")
    
    target_col = 'Churn'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: 80% train, 20% temp (for val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: 50% of temp for val, 50% for test (making each 10% of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train set size: {len(X_train)} rows")
    print(f"Validation set size: {len(X_val)} rows")
    print(f"Test set size: {len(X_test)} rows")

    # --- 4. Fit Preprocessor on Training Data ONLY (CRITICAL STEP) ---
    print("\nFitting the data preprocessor on the TRAINING data to prevent data leakage...")
    preprocessor = ChurnDataPreprocessor()
    # We pass the features (X_train) and tell the preprocessor to ignore the target if it sees it
    preprocessor.fit(X_train, target_col=None) # We already separated y, so no target to ignore
    print("Preprocessor fitting complete.")

    # --- 5. Transform All Data Splits ---
    print("Transforming train, validation, and test sets using the fitted preprocessor...")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    print("All sets transformed successfully.")

    # --- 6. Save All Processed Data and the Preprocessor Artifact ---
    print("\nSaving all processed data files and the preprocessor artifact...")
    
    # Save processed feature sets
    X_train_processed.to_csv(processed_data_path / "X_train.csv", index=False)
    X_val_processed.to_csv(processed_data_path / "X_val.csv", index=False)
    X_test_processed.to_csv(processed_data_path / "X_test.csv", index=False)

    # Save target sets
    y_train.to_csv(processed_data_path / "y_train.csv", index=False, header=True)
    y_val.to_csv(processed_data_path / "y_val.csv", index=False, header=True)
    y_test.to_csv(processed_data_path / "y_test.csv", index=False, header=True)
    
    # Save the crucial fitted preprocessor
    preprocessor_save_path = artifacts_path / "preprocessor.joblib"
    preprocessor.save(preprocessor_save_path)
    
    print("\n----------------------------------------------------")
    print("SUCCESS: Data pipeline completed successfully.")
    print(f"Processed data saved in: {processed_data_path}")
    print(f"Fitted preprocessor saved to: {preprocessor_save_path}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    run_data_pipeline()