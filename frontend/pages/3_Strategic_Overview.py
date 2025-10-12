import streamlit as st
import pandas as pd
import plotly.express as px
from middleware.requestHandler import APIRequestHandler

st.set_page_config(page_title="Strategic Overview", layout="wide")
api = APIRequestHandler("http://localhost:8000")

st.title("🎯 Strategic Segment Analysis")
st.markdown("Use the filters to dynamically segment your customer base and uncover key insights.")

# --- 1. Interactive Filters (Connected to the REAL Backend Endpoint) ---
st.sidebar.header("Interactive Segment Filters")

# These filters now directly match the `/analytics/segment` endpoint parameters.
risk_range = st.sidebar.slider(
    "Filter by Churn Risk Score",
    min_value=0.0, max_value=1.0, value=(0.5, 1.0) # Default to high-risk customers
)

age_range = st.sidebar.slider(
    "Filter by Customer Age",
    min_value=18, max_value=110, value=(18, 100)
)

tenure_range = st.sidebar.slider(
    "Filter by Tenure (Days)",
    min_value=0, max_value=20, value=(0, 10000)
)

income_range = st.sidebar.slider(
    "Filter by Annual Income ($)",
    min_value=0, max_value=400000, value=(0, 200000), step=1000
)

# --- 2. Unified API Call ---
# We make ONE powerful call to the endpoint you already built.
@st.cache_data(ttl=60) # Cache the data for 60 seconds to improve performance
def fetch_segment_data(min_risk, max_risk, min_age, max_age, min_tenure, max_tenure, min_income, max_income):
    params = {
        "min_risk": min_risk, "max_risk": max_risk,
        "min_age": min_age, "max_age": max_age,
        "min_tenure": min_tenure, "max_tenure": max_tenure,
        "min_income": min_income, "max_income": max_income,
    }
    try:
        response = api.get("/analytics/segment", params=params)
        if response and response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
    return None

data = fetch_segment_data(
    risk_range[0], risk_range[1],
    age_range[0], age_range[1],
    tenure_range[0], tenure_range[1],
    income_range[0], income_range[1]
)

if not data:
    st.error("Could not fetch analytics data from the backend. Is the server running?")
else:
    # --- 3. High-Impact KPI Metrics (The "Impressive" Part) ---
    summary = data.get('summary', {})
    st.subheader("Segment Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Segment Size (Customers)", f"{summary.get('segment_size', 0):,}")
    col2.metric("Average Churn Risk", f"{summary.get('average_churn_risk', 0):.1%}")
    col3.metric("Top Risk Factor", data.get('top_risk_factors', [{}])[0].get('factor', 'N/A'))

    st.divider()

    # --- 4. Insightful Visualizations ---
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Top Risk Factors for this Segment")
        factors = data.get('top_risk_factors', [])
        if factors:
            df_factors = pd.DataFrame(factors)
            fig_factors = px.bar(df_factors, x='count', y='factor',
                                 orientation='h', title="Most Common Churn Drivers",
                                 labels={'factor': 'Risk Factor', 'count': 'Number of Customers'},
                                 color='count', color_continuous_scale='OrRd')
            fig_factors.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_factors, use_container_width=True)
        else:
            st.info("No risk factor data for this segment.")

    with col_b:
        st.subheader("Churn Risk Distribution")
        customers = data.get('customers', [])
        if customers:
            df_cust = pd.DataFrame(customers)
            fig_hist = px.histogram(df_cust, x='churn_probability', nbins=20,
                                    title="How Risk is Distributed in the Segment",
                                    labels={'churn_probability': 'Churn Probability Score'})
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No customer data for this segment.")

    # --- 5. Actionable Data Table ---
    st.subheader("Customers in this Segment")
    if customers:
        # Use the same DataFrame as the histogram above
        st.dataframe(df_cust, use_container_width=True)
    else:
        st.info("No customers match the selected filter criteria.")