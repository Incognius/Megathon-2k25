import matplotlib.pyplot as plt
import shap
import os

# Add this to the top to handle the 'src' import issue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.churnxai.explain import ChurnExplainer

def main():
    """
    Runs the explainer and generates a SHAP summary plot.
    """
    print("--- Running Explanation Generation ---")
    
    # Initialize our explainer with the champion model
    explainer = ChurnExplainer("artifacts/models/champion_model_xgb.joblib")
    
    # Generate explanations for a sample from the test set
    shap_values, X_sample = explainer.explain_instances("data/processed/X_test.csv")

    # --- Generate and Save SHAP Summary Plot ---
    print("Generating SHAP summary plot...")
    
    # The summary plot shows the most important features and their impact.
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    
    # Save the plot
    output_dir = "artifacts/reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "4_shap_summary_plot.png")
    
    # Use bbox_inches='tight' to prevent labels from being cut off
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSHAP summary plot saved to {plot_path}")
    print("This plot shows the most influential features for predicting churn.")
    print("Red dots are high feature values, blue are low. A positive SHAP value pushes the prediction towards churn.")

if __name__ == "__main__":
    main()