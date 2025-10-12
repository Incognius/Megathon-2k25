import shap
import pandas as pd
import joblib

class ChurnExplainer:
    """
    A class to generate SHAP explanations for the champion churn model.
    """
    def __init__(self, model_path: str):
        """
        Initializes the explainer by loading the champion model.
        
        Args:
            model_path (str): Path to the saved champion model (e.g., champion_model_xgb.joblib).
        """
        print("--- Initializing Churn Explainer ---")
        self.model = joblib.load(model_path)
        # For tree models, the TreeExplainer is highly optimized.
        self.explainer = shap.TreeExplainer(self.model)
        print("Champion model and SHAP TreeExplainer loaded.")

    def explain_instances(self, X_data_path: str, n_instances: int = 200):
        """
        Generates SHAP values for a sample of instances from a given dataset.

        Args:
            X_data_path (str): Path to the preprocessed data to explain (e.g., X_test.csv).
            n_instances (int): The number of random instances to explain.

        Returns:
            shap_values: The calculated SHAP values.
            X_sample: The DataFrame sample for which explanations were generated.
        """
        print(f"--- Generating SHAP explanations for {n_instances} instances ---")
        
        X_df = pd.read_csv(X_data_path)
        
        # Take a random sample for explanation. In a real app, this could be specific customers.
        X_sample = X_df.sample(n_instances, random_state=42)
        
        # Calculate SHAP values. For binary classification, this returns a single array of values.
        shap_values = self.explainer.shap_values(X_sample)
        
        print("SHAP values calculated successfully.")
        return shap_values, X_sample
