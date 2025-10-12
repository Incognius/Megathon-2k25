import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import sys

# Add the project root to Python's path at the VERY TOP
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.churnxai.preprocess import ChurnDataPreprocessor

st.set_page_config(page_title="Strategy Sandbox", layout="wide")

# --- Load Model, Preprocessor, and Data ONCE ---
@st.cache_resource(ttl=3600)
def load_artifacts():
    """Loads all necessary artifacts into memory."""
    try:
        model = joblib.load(project_root / "artifacts/models/champion_model_xgb.joblib")
        preprocessor = joblib.load(project_root / "artifacts/models/preprocessor.joblib")
        
        # Load a large, random sample to prevent memory crashes.
        raw_data = pd.read_csv(project_root / "data/raw/autoinsurance_churn.csv").sample(n=50000, random_state=42)
        
        # --- THE FIX: The return statement is now on its own, correctly indented line. ---
        return model, preprocessor, raw_data

    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Could not load a required artifact: {e}. Please ensure all pipeline scripts have been run.")
        return None, None, None

model, preprocessor, raw_df = load_artifacts()

st.title("💡 Strategy Sandbox: A Live 'What-If' Engine")
st.markdown("Design a retention campaign and simulate its real-time impact on churn risk.")

# --- Main App Logic ---
if model and preprocessor and raw_df is not None:
    st.sidebar.header("1. Define Your Target Segment")
    age_range = st.sidebar.slider("Target Age Range", 18, 100, (18, 40))
    tenure_range_days = st.sidebar.slider("Target Tenure (Days)", 0, 10000, (0, 730))
    income_range = st.sidebar.slider("Target Annual Income ($)", 0, 200000, (0, 50000), step=1000)

    st.sidebar.header("2. Define Your Retention Action")
    premium_discount_pct = st.sidebar.slider("Offer Premium Discount (%)", 0, 50, 10)

    segment_df = raw_df[
        (raw_df['age_in_years'].between(age_range[0], age_range[1])) &
        (raw_df['days_tenure'].between(tenure_range_days[0], tenure_range_days[1])) &
        (raw_df['income'].between(income_range[0], income_range[1]))
    ].copy()

    if st.button("🚀 Run Live Simulation", use_container_width=True) and not segment_df.empty:
        with st.spinner("Running simulation... This may take a moment."):
            # "BEFORE" ANALYSIS
            X_original = preprocessor.transform(segment_df)
            X_original = X_original[model.get_booster().feature_names]
            risk_original = model.predict_proba(X_original)[:, 1]

            # "AFTER" ANALYSIS (The Simulation)
            segment_df_modified = segment_df.copy()
            segment_df_modified['curr_ann_amt'] *= (1 - premium_discount_pct / 100)
            
            X_modified = preprocessor.transform(segment_df_modified)
            X_modified = X_modified[model.get_booster().feature_names]
            risk_new = model.predict_proba(X_modified)[:, 1]

            # VISUALIZE THE IMPACT
            st.subheader("Simulation Impact Analysis")
            
            customers_saved = ((risk_original > 0.5) & (risk_new <= 0.5)).sum()
            cost_of_campaign = (segment_df['curr_ann_amt'] * (premium_discount_pct / 100)).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Target Segment Size", f"{len(segment_df):,} Customers")
            col2.metric("Potential Customers Saved", f"{customers_saved:,}", help="Customers whose risk score dropped from >50% to <50%.")
            col3.metric("Estimated Campaign Cost", f"${cost_of_campaign:,.0f}", help="Total annual cost of the applied discount for this segment.")

            st.markdown("#### Churn Risk Distribution: Before vs. After")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=risk_original, name='Original Risk', marker_color='#EF553B', opacity=0.75))
            fig.add_trace(go.Histogram(x=risk_new, name='New Risk (After Discount)', marker_color='#00CC96', opacity=0.75))
            fig.update_layout(barmode='overlay', title_text='Comparing Risk Distributions', xaxis_title_text='Churn Probability', yaxis_title_text='Number of Customers')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Simulation complete! The chart above shows how the discount shifted the churn risk for the entire segment.")

    elif segment_df.empty:
        st.warning("No customers match the selected filter criteria. Please adjust the filters.")

else:
    st.error("Could not load model artifacts. The sandbox is disabled.")