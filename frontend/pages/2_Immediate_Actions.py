import streamlit as st
import google.generativeai as genai
from middleware.requestHandler import APIRequestHandler

# --- Page Configuration ---
st.set_page_config(page_title="Immediate Actions", layout="wide")
api = APIRequestHandler("http://localhost:8000")

# --- Configure Gemini AI Agent ---
try:
    # Use st.secrets for secure key management in production
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    st.session_state.gemini_enabled = True
except Exception:
    st.session_state.gemini_enabled = False
    gemini_model = None

# --- Initialize Session State ---
if 'selected_customer_id' not in st.session_state:
    st.session_state.selected_customer_id = None
if 'actionable_customers' not in st.session_state:
    st.session_state.actionable_customers = None

def go_to_customer(customer_id):
    """Callback to switch to the detail view."""
    st.session_state.selected_customer_id = customer_id

def get_ai_recommendation(customer_data):
    """Calls the Gemini API to generate a retention script."""
    if not st.session_state.gemini_enabled or not gemini_model:
        return "AI Agent is disabled. Please configure the Gemini API key in `.streamlit/secrets.toml`."

    prompt = f"""
    You are an expert retention agent at an auto insurance company.
    A customer is at high risk of churning. Here is their data:
    - Churn Probability: {customer_data['risk_score']:.1%}
    - Top churn risk factors: {customer_data['reason_codes']}
    - Customer Tenure: {customer_data['customer_data'].get('tenure', 'N/A')} days
    - Customer Annual Premium: ${customer_data['customer_data'].get('premium', 0):,.2f}

    Based on this data, generate a concise, actionable script for a retention agent to use when calling this customer.
    The script should:
    1. Acknowledge their potential concerns based on the risk factors.
    2. Propose a specific, targeted solution.
    3. Be empathetic and professional.

    Format the output in Markdown. Start with a "### Recommended Agent Script" header.
    """
    try:
        with st.spinner("🤖 AI Agent is crafting a personalized retention script..."):
            response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI recommendation: {e}"

# --- RENDER FUNCTIONS ---

def render_actionables_list():
    """Renders the list of high-risk customers to take action on."""
    st.subheader("High-Risk Customers Requiring Immediate Action")
    
    # Add a slider to control the risk threshold
    risk_threshold = st.slider("Set Churn Risk Threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.05)

    # Fetch data from the backend using the selected threshold
    with st.spinner("Fetching high-risk customers from ML backend..."):
        resp = api.get(f"/actions/immediate?risk_threshold={risk_threshold}")
        if resp and resp.status_code == 200:
            st.session_state.actionable_customers = resp.json().get('actions', [])
        else:
            st.session_state.actionable_customers = []

    if not st.session_state.actionable_customers:
        st.warning("No customers found above the selected risk threshold.")
        return

    # Display each customer in the action list
    for cust in st.session_state.actionable_customers:
        col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
        with col1:
            st.write(f"**{cust['name']}**")
        with col2:
            st.progress(cust['risk_score'], text=f"Risk: {cust['risk_score']:.1%}")
        with col3:
            st.info(f"Top Reason: **{cust['reason_codes'].split(',')[0].strip()}**")
        with col4:
            st.button("Details", key=f"action_{cust.get('id')}", on_click=go_to_customer, args=(cust.get('id'),))

def render_customer_detail_view():
    """Renders the detailed action plan and AI script for a single customer."""
    cust_id = st.session_state.selected_customer_id
    
    with st.spinner(f"Fetching details for customer {cust_id}..."):
        resp = api.get(f"/customer/{cust_id}")

    if st.button("← Back to Action List"):
        st.session_state.selected_customer_id = None
        st.rerun()

    if not resp or resp.status_code != 200:
        st.error(f"Could not fetch details for customer {cust_id}.")
        return

    data = resp.json()
    st.header(f"Action Plan for {data['customer_name']}")

    # Display key metrics
    col1, col2 = st.columns(2)
    col1.metric("Churn Risk Score", f"{data['risk_score']:.1%}", delta_color="inverse")
    col2.metric("Top Risk Factor", data.get('reason_codes', 'N/A').split(',')[0].strip())

    st.divider()

    # --- AI Agent Section ---
    st.subheader("🤖 AI Retention Co-Pilot")
    if not st.session_state.gemini_enabled:
        st.error("Gemini API key not found in `.streamlit/secrets.toml`. AI Agent is disabled.")
    else:
        recommendation_script = get_ai_recommendation(data)
        st.markdown(recommendation_script)

# --- MAIN PAGE LOGIC ---
st.title("Immediate Actions")

# This logic decides whether to show the list or the detail view
if st.session_state.selected_customer_id:
    render_customer_detail_view()
else:
    render_actionables_list()
