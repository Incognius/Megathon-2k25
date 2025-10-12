import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

def generate_model_report():
    """
    Loads the champion model and test data to generate a comprehensive
    visual report including confusion matrix, feature importance, and ROC curve.
    """
    print("--- Generating Visual Model Report ---")
    project_root = Path(__file__).resolve().parents[1]
    artifacts_path = project_root / "artifacts"
    data_path = project_root / "data/processed"

    # --- 1. Load Model and Test Data ---
    try:
        print("Loading champion model and test data...")
        model = joblib.load(artifacts_path / "models/champion_model_xgb.joblib")
        X_test = pd.read_csv(data_path / "X_test.csv")
        y_test = pd.read_csv(data_path / "y_test.csv").values.ravel()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required file: {e}. Ensure the data split and model training scripts have run.")
        return

    # --- 2. Make Predictions on Test Data ---
    print("Making predictions on the test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # --- 3. Create a Multi-Panel Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('XGBoost Champion Model Performance Report', fontsize=20, weight='bold')
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Panel 1: Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'],
                annot_kws={"size": 16})
    axes[0].set_title('Confusion Matrix', fontsize=16, weight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)

    # --- Panel 2: Feature Importance ---
    print("Generating Feature Importance plot...")
    # Get top 15 features for clarity
    importance = pd.Series(model.feature_importances_, index=X_test.columns).nlargest(15)
    sns.barplot(x=importance.values, y=importance.index, ax=axes[1], palette='viridis')
    axes[1].set_title('Top 15 Feature Importances', fontsize=16, weight='bold')
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_ylabel('Features', fontsize=12)

    # --- Panel 3: ROC Curve ---
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('False Positive Rate', fontsize=12)
    axes[2].set_ylabel('True Positive Rate', fontsize=12)
    axes[2].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, weight='bold')
    axes[2].legend(loc="lower right", fontsize=12)
    axes[2].grid(True)

    # --- 4. Save the Final Report Image ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    report_path = artifacts_path / "reports"
    report_path.mkdir(exist_ok=True)
    final_image_path = report_path / "model_performance_report.png"
    fig.savefig(final_image_path, dpi=150)
    
    print("\n-----------------------------------------")
    print(f"SUCCESS: Model report saved to:\n{final_image_path}")
    print("-----------------------------------------")

if __name__ == "__main__":
    generate_model_report()