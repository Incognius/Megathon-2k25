import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(csv_path: str, target_col: str):
    """
    Loads data, performs basic EDA, and saves summary plots.
    
    Args:
        csv_path (str): Path to the raw CSV data file.
        target_col (str): The name of the target column for churn.
    """
    # --- 1. Setup and Load Data ---
    print(f"Starting EDA for dataset: {csv_path}")
    output_dir = "artifacts/reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Efficiently load data, especially for 1.6M rows
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return

    # --- 2. Basic Information ---
    print("\n--- Dataset Info ---")
    df.info(verbose=True, show_counts=True)
    
    print("\n--- Dataset Head ---")
    print(df.head())
    
    print("\n--- Numerical Summary ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # --- 3. Visualizations ---
    print(f"\n--- Generating and saving plots to {output_dir} ---")
    
    # Plot 1: Target Variable Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df, hue=target_col, palette="viridis", legend=False)
    plt.title(f'Distribution of Target Variable: {target_col}')
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    churn_dist_path = os.path.join(output_dir, '1_churn_distribution.png')
    plt.savefig(churn_dist_path)
    plt.close()
    print(f"Saved churn distribution plot to {churn_dist_path}")

    # Plot 2: Correlation Heatmap for Numerical Features
    plt.figure(figsize=(16, 12))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    corr_matrix_path = os.path.join(output_dir, '2_correlation_matrix.png')
    plt.savefig(corr_matrix_path)
    plt.close()
    print(f"Saved correlation matrix to {corr_matrix_path}")
    
    print("\nEDA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis for the Churn Prediction project.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target-col", type=str, default="Churn", help="Name of the target column in the dataset.")
    
    args = parser.parse_args()
    
    perform_eda(csv_path=args.csv, target_col=args.target_col)
