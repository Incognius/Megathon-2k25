import streamlit as st
from middleware.requestHandler import APIRequestHandler

st.set_page_config(page_title="Dashboard", layout="wide")

# --- Initialize session state ---
if 'selected_customer_id' not in st.session_state:
    st.session_state.selected_customer_id = None
if 'customers' not in st.session_state:
    st.session_state.customers = None

api = APIRequestHandler("http://localhost:8000")

def go_to_customer(customer_id):
    st.session_state.selected_customer_id = customer_id

# --- THE "INSIGHT DICTIONARY" FOR DEEP EXPLAINABILITY ---
# This dictionary provides a title and a business-focused explanation for each raw feature name.
# --- THE "INSIGHT DICTIONARY" FOR DEEP EXPLAINABILITY ---
REASON_CODE_INSIGHTS = {
    "curr_ann_amt": {
        "title": "High Annual Premium",
        "explanation": "The customer's annual premium is significantly high. This is a primary driver for customers to shop around for more competitive pricing from other insurers."
    },
    "days_tenure": {
        "title": "Short Customer Tenure",
        "explanation": "This is a relatively new customer. Newer customers often have weaker brand loyalty and are more likely to switch providers after their initial term."
    },
    "age_in_years": {
        "title": "Customer Age Group",
        "explanation": "The model has identified that customers in this age bracket (often younger) tend to be more price-sensitive and less brand-loyal, frequently seeking better deals."
    },
    "income": {
        "title": "Income Level",
        "explanation": "Customers with lower income are more sensitive to premium costs. A recent rate increase or a change in their financial situation could trigger a search for a cheaper policy."
    },
    "premium_to_income_ratio": {
        "title": "High Premium-to-Income Ratio",
        "explanation": "The insurance premium represents a large percentage of this customer's income. This financial pressure makes them extremely likely to churn if a lower-cost alternative is available."
    },
    "geo_cluster": {
        "title": "Geographic Risk Zone",
        "explanation": "The customer is located in a geographic cluster where the model has detected a higher-than-average churn rate, possibly due to regional competition or demographic trends."
    },
    "policy_type": {
        "title": "Policy Type",
        "explanation": "The specific type of policy this customer holds is associated with a higher churn risk in the historical data."
    },
    "marital_status": {
        "title": "Marital Status",
        "explanation": "The model has learned that an individual's marital status can influence their churn risk. Customers who are single or have undergone a recent life change (e.g., divorce) may be re-evaluating their finances and are often more likely to seek new insurance providers."
    },
}

# --- RENDER FUNCTIONS ---
def render_customer_list_view():
    st.subheader("Customer Churn Risk Rankings")
    
    if st.session_state.customers is None:
        with st.spinner("Fetching customer data from ML backend..."):
            resp = api.get("/customers?limit=100")
            if resp and resp.status_code == 200:
                st.session_state.customers = resp.json().get('customers', [])
            else:
                st.session_state.customers = []
    
    if not st.session_state.customers:
        st.warning("Could not fetch customer data or no customers found.")
        return

    for cust in st.session_state.customers:
        col1, col2, col3 = st.columns([4, 4, 1])
        with col1:
            st.write(f"**{cust['name']}**")
        with col2:
            st.progress(cust['risk_score'], text=f"Churn Probability: {cust['risk_score']:.1%}")
        with col3:
            st.button("Details", key=cust.get("id"), on_click=go_to_customer, args=(cust.get("id"),))

def render_customer_detail_view():
    cust_id = st.session_state.selected_customer_id
    
    with st.spinner(f"Fetching ML-powered details for customer {cust_id}..."):
        resp = api.get(f"/customer/{cust_id}")

    if st.button("← Back to List"):
        st.session_state.selected_customer_id = None
        st.rerun()

    if not resp or resp.status_code != 200:
        st.error(f"Could not fetch details for customer {cust_id}.")
        return

    data = resp.json()
    st.header(f"Analysis for {data['customer_name']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Risk Score", f"{data['risk_score']:.1%}", delta_color="inverse")
    col2.metric("Tenure (Days)", f"{data['customer_data'].get('tenure', 'N/A'):.0f}")
    col3.metric("Annual Premium", f"${data['customer_data'].get('premium', 0):,.2f}")

    st.divider()
    
    # --- THIS IS THE FULLY UPGRADED EXPLANATION SECTION ---
    st.subheader("Top Reasons for Churn Risk")
    reasons_raw = data.get('reason_codes', 'N/A').split(',')
    
    if reasons_raw[0] == 'N/A':
        st.write("No specific reason codes available for this customer.")
    else:
        for reason in reasons_raw:
            cleaned_reason = reason.strip()
            # Look up the insight from our dictionary
            insight = REASON_CODE_INSIGHTS.get(cleaned_reason)
            
            if insight:
                # Display the title and the full explanation
                st.markdown(f"##### 🔹 {insight['title']}")
                st.write(insight['explanation'])
            else:
                # Safe fallback if the reason is not in our dictionary
                st.markdown(f"##### 🔹 {cleaned_reason}")
                st.write("No detailed explanation available for this factor.")

    # Display raw data for reference
    with st.expander("Show Full Customer Data"):
        st.json(data['customer_data'])

# --- MAIN LOGIC ---
st.title("🛡️ Guardian Dashboard")

if st.session_state.selected_customer_id:
    render_customer_detail_view()
else:
    render_customer_list_view()