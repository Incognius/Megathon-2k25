import pandas as pd
import joblib
import shap
import logging
from pathlib import Path

# --- CONFIGURATION ---
TOP_N_CUSTOMERS_TO_EXPLAIN = 1000  # Explain only the top 1000 highest-risk customers. FAST.

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_final_predictions():
    """
    Identifies the highest-risk customers, runs SHAP explanations ONLY for them,
    merges original data, and saves a fast, focused, and complete results file.
    """
    project_root = Path(__file__).resolve().parents[1]
    artifacts_path = project_root / "artifacts"
    data_path = project_root / "data"

    # --- 1. Load Model and Preprocessor ---
    try:
        logging.info("Loading model and preprocessor...")
        model = joblib.load(artifacts_path / "models/champion_model_xgb.joblib")
        preprocessor = joblib.load(artifacts_path / "models/preprocessor.joblib")
    except FileNotFoundError as e:
        logging.error(f"FATAL: Model/preprocessor not found: {e}. Run training first.")
        return

    # --- 2. Load Raw Data ---
    logging.info("Loading original raw dataset...")
    try:
        full_df = pd.read_csv(data_path / "raw/autoinsurance_churn.csv")
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"FATAL: Could not load raw data: {e}")
        return

    # --- 3. Preprocess All Data ---
    logging.info("Preprocessing all data to get predictions...")
    transform_output = preprocessor.transform(full_df.copy())
    X_processed = transform_output[0] if isinstance(transform_output, tuple) else transform_output
    X_processed = X_processed[model.get_booster().feature_names]

    # --- 4. Identify Top N Highest-Risk Customers ---
    logging.info(f"Running predictions for all customers to find the top {TOP_N_CUSTOMERS_TO_EXPLAIN} risks...")
    all_probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Create a temporary DataFrame with IDs and probabilities
    risk_df = pd.DataFrame({
        'individual_id': full_df['individual_id'],
        'churn_probability': all_probabilities,
        'original_index': full_df.index # Keep track of original position
    })

    # Get the top N highest-risk customers
    top_risk_df = risk_df.nlargest(TOP_N_CUSTOMERS_TO_EXPLAIN, 'churn_probability')
    top_risk_indices = top_risk_df['original_index'].values

    # --- 5. Run SHAP ONLY on the High-Risk Segment (FAST!) ---
    logging.info(f"Running SHAP explanations ONLY for the top {TOP_N_CUSTOMERS_TO_EXPLAIN} customers...")
    
    # Filter the processed data to only the top N
    X_processed_top_risk = X_processed.loc[top_risk_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values_top_risk = explainer.shap_values(X_processed_top_risk)
    
    # Map SHAP values to feature names
    feature_names = X_processed_top_risk.columns
    top_reasons = []
    for i in range(len(shap_values_top_risk)):
        top_indices = pd.Series(abs(shap_values_top_risk[i])).nlargest(3).index
        reasons = [feature_names[j] for j in top_indices]
        top_reasons.append(", ".join(reasons))
        
    top_risk_df['reason_codes'] = top_reasons

    # --- 6. Merge and Save the Final, Focused File ---
    logging.info("Merging human-readable data for the top-risk segment...")
    
    # Filter the original dataframe to match the top N
    original_data_top_risk = full_df.loc[top_risk_indices]

    # Merge the SHAP results with the original data
    enriched_results_df = pd.merge(
        left=top_risk_df[['individual_id', 'churn_probability', 'reason_codes']],
        right=original_data_top_risk,
        on='individual_id',
        how='left'
    )
    
    # Final rename to match the backend API
    enriched_results_df = enriched_results_df.rename(columns={'individual_id': 'customer_id'})

    # Save the file
    output_path = artifacts_path / "results/predictions_with_explanations.csv"
    output_path.parent.mkdir(exist_ok=True)
    enriched_results_df.to_csv(output_path, index=False)
    
    logging.info(f"SUCCESS: A focused dataset of the top {TOP_N_CUSTOMERS_TO_EXPLAIN} highest-risk customers has been saved to {output_path}.")

if __name__ == "__main__":
    run_final_predictions()