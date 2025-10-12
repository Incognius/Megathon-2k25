import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import os
import warnings

class ChurnDataPreprocessor:
    """
    A class to handle all preprocessing for the churn dataset.
    Version 4: A robust, unified pipeline.
    - Drops constant/leakage columns ('acct_suspd_date', 'state').
    - Engineers all features internally (home value, age bins, ratios, clusters).
    - Correctly handles fitting and transforming data with all features.
    """
    def __init__(self, n_clusters: int = 12):
        self.n_clusters = n_clusters
        self.imputation_values = {}
        self.county_geo_medians = {}
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.categorical_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_order = []
        
        self.cols_to_drop = ['acct_suspd_date', 'state', 'cust_orig_date', 'date_of_birth']

    def _engineer_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Private method to create all new features."""
        print("Engineering all features...")
        df = df.copy()

        # 1. Home Market Value -> Lower/Upper Bounds
        if 'home_market_value' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bounds = df['home_market_value'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)
            df['home_value_lower'] = bounds[0]
            df['home_value_upper'] = bounds[1]
        
       

        # 3. Premium-to-Income Ratio (handle potential missing income before ratio)
        df['premium_to_income_ratio'] = df['curr_ann_amt'] / (df['income'].fillna(0) + 1e-6)

        # 4. Geographic Clustering with K-Means
        if 'latitude' in df.columns and 'longitude' in df.columns:
            geo_features = df[['latitude', 'longitude']].copy()
            # Impute geo features specifically for K-Means fitting/prediction
            for col in ['latitude', 'longitude']:
                if col in self.imputation_values:
                    geo_features[col] = geo_features[col].fillna(self.imputation_values[col])
                else: # Fallback for the very first fit
                    geo_features[col] = geo_features[col].fillna(geo_features[col].median())
            
            if is_train:
                print(f"Fitting K-Means with {self.n_clusters} clusters...")
                self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                df['geo_cluster'] = self.kmeans_model.fit_predict(geo_features)
            else:
                df['geo_cluster'] = self.kmeans_model.predict(geo_features)
            
            df['geo_cluster'] = 'cluster_' + df['geo_cluster'].astype(str)
            
        return df

    def fit(self, df: pd.DataFrame, target_col: str):
        """Learns all transformation parameters from the training data."""
        print("--- Fitting Preprocessor ---")
        
        df_clean = df.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Learn county-based geo medians first
        if 'county' in df_clean.columns:
            self.county_geo_medians = df_clean.groupby('county')[['latitude', 'longitude']].median().to_dict('index')

        # Engineer features for the training set
        df_engineered = self._engineer_features(df_clean, is_train=True)

        # Identify column types FROM THE ENGINEERED DATAFRAME
        all_cols = df_engineered.drop(columns=[target_col], errors='ignore').columns.tolist()
        id_cols = [col for col in all_cols if 'id' in col.lower()]
        
        self.categorical_cols = df_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = [col for col in all_cols if col not in self.categorical_cols and col not in id_cols]

        print(f"Final numeric features ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Final categorical features ({len(self.categorical_cols)}): {self.categorical_cols}")

        # Learn imputation values and fit scaler/encoders
        df_imputed = self._impute(df_engineered)
        
        for col in self.numeric_cols:
            self.imputation_values[col] = df_imputed[col].median()
        for col in self.categorical_cols:
            self.imputation_values[col] = 'Unknown'
        
        df_imputed_final = self._impute(df_engineered)

        if self.numeric_cols:
            self.scaler.fit(df_imputed_final[self.numeric_cols])

        for col in self.categorical_cols:
            le = LabelEncoder()
            all_values = pd.concat([df_imputed_final[col].dropna(), pd.Series(['Unknown'])]).astype(str).unique()
            le.fit(all_values)
            self.categorical_encoders[col] = le
            
        self.feature_order = self.numeric_cols + self.categorical_cols
        print("--- Preprocessor fitting complete. ---")
        return self

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to apply all learned imputation strategies."""
        df = df.copy()
        if self.county_geo_medians:
            for county, medians in self.county_geo_medians.items():
                mask = (df['county'] == county)
                df.loc[mask, 'latitude'] = df.loc[mask, 'latitude'].fillna(medians['latitude'])
                df.loc[mask, 'longitude'] = df.loc[mask, 'longitude'].fillna(medians['longitude'])
        
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all learned transformations to new data."""
        print("--- Transforming data ---")
        df_trans = df.drop(columns=self.cols_to_drop, errors='ignore')
        df_trans = self._engineer_features(df_trans, is_train=False)
        df_trans = self._impute(df_trans)

        df_numeric = pd.DataFrame(index=df_trans.index)
        if self.numeric_cols:
            df_scaled = self.scaler.transform(df_trans[self.numeric_cols].fillna(0)) # Final fillna for safety
            df_numeric = pd.DataFrame(df_scaled, columns=self.numeric_cols, index=df_trans.index)
        
        df_categorical = pd.DataFrame(index=df_trans.index)
        for col, encoder in self.categorical_encoders.items():
            df_col_str = df_trans[col].astype(str)
            known_values = encoder.classes_
            transformed_col = df_col_str.apply(lambda x: x if x in known_values else 'Unknown')
            df_categorical[col] = encoder.transform(transformed_col)

        X = pd.concat([df_numeric, df_categorical], axis=1)
        return X[self.feature_order]
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)
