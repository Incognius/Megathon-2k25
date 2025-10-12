import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.churnxai.preprocess import ChurnDataPreprocessor

def find_k(csv_path: str, max_k: int = 50):
    """
    Performs the Elbow Method to find the optimal number of clusters (k).
    """
    print("--- Finding Optimal K for K-Means ---")
    
    # --- 1. Load and prepare a sample of the data for speed ---
    df = pd.read_csv(csv_path, usecols=['latitude', 'longitude', 'county'])
    df_sample = df.dropna().sample(n=50000, random_state=42) # Use a large sample
    
    # --- 2. Impute missing values simply for this task ---
    # We don't need the full preprocessor, just filled lat/lon
    df_sample['latitude'].fillna(df_sample['latitude'].median(), inplace=True)
    df_sample['longitude'].fillna(df_sample['longitude'].median(), inplace=True)
    
    geo_features = df_sample[['latitude', 'longitude']]

    # --- 3. Run K-Means for a range of k and calculate inertia ---
    inertias = []
    k_range = range(2, max_k + 1)
    print(f"Testing k from {min(k_range)} to {max(k_range)}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(geo_features)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}, Inertia={kmeans.inertia_:.2f}")

    # --- 4. Plot the results ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(k_range)
    plt.grid(True)
    
    output_dir = "artifacts/reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, '3_kmeans_elbow_plot.png')
    plt.savefig(plot_path)
    
    print(f"\nElbow plot saved to {plot_path}")
    print("Inspect the plot to find the 'elbow' point. This is your optimal k.")
    print("It's the point where the rate of decrease in inertia sharply slows down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal k for K-Means clustering using the Elbow Method.")
    parser.add_argument("--csv", type=str, default="data/raw/autoinsurance_churn.csv", help="Path to the input CSV file.")
    args = parser.parse_args()
    find_k(csv_path=args.csv)