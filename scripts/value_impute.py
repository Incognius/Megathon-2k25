import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def intelligent_impute(df):
    """
    Performs sophisticated, context-aware imputation on the dataset.

    Args:
        df (pd.DataFrame): The raw dataframe with missing values.

    Returns:
        pd.DataFrame: The dataframe with missing values cleverly imputed.
    """
    print("--- Starting Intelligent Imputation ---")
    print("Missing values BEFORE imputation:")
    print(df.isnull().sum())

    # --- 1. Impute Longitude/Latitude using City Centroid ---
    # As you suggested, we fill missing coordinates with the mean of their city.
    print("\nStep 1: Imputing missing coordinates with city centroids...")
    df['latitude'] = df.groupby('city')['latitude'].transform(lambda x: x.fillna(x.mean()))
    df['longitude'] = df.groupby('city')['longitude'].transform(lambda x: x.fillna(x.mean()))

    # --- 2. Impute City using Geographic Neighbors (KNN) ---
    # For missing cities, we find the nearest neighbors based on coordinates.
    print("Step 2: Imputing missing cities using K-Nearest Neighbors on coordinates...")
    
    # Temporarily encode cities to numbers for the imputer
    df['city_encoded'] = df['city'].astype('category').cat.codes
    df['city_encoded'] = df['city_encoded'].replace(-1, np.nan) # Set missing to NaN

    # Use KNN to impute the missing city codes based on lat/lon
    knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
    impute_cols = ['latitude', 'longitude', 'city_encoded']
    df[impute_cols] = knn_imputer.fit_transform(df[impute_cols])
    
    # Round the imputed codes and convert back to city names
    df['city_encoded'] = df['city_encoded'].round().astype(int)
    city_mapping = dict(enumerate(df['city'].astype('category').cat.categories))
    df['city'] = df['city_encoded'].map(city_mapping)
    df = df.drop(columns='city_encoded')


    # --- 3. Impute Home Market Value using County Median ---
    # We fill missing home values with the median value of the customer's county.
    print("Step 3: Imputing home market value with county-level medians...")
    df['home_market_value'] = df.groupby('county')['home_market_value'].transform(lambda x: x.fillna(x.median()))
    
    # If any counties are fully NaN, fill with global median as a fallback
    if df['home_market_value'].isnull().any():
        df['home_market_value'] = df['home_market_value'].fillna(df['home_market_value'].median())

    print("\n--- Imputation Complete ---")
    print("Missing values AFTER imputation:")
    print(df.isnull().sum())
    
    return df

# --- Example Usage ---
if _name_ == '_main_':
    # Create a sample DataFrame with missing values to demonstrate the process
    samples = pd.read_csv('./datasets/autoinsurance_churn.csv')

    # Run the intelligent imputation
    cleaned_df = intelligent_impute(samples.copy())

    print("\n--- Final Cleaned DataFrame ---")
    print(cleaned_df)
    
    # Save the cleaned data to a new file
    # cleaned_df.to_csv("cleaned_customer_data.csv", index=False)
