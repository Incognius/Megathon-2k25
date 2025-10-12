import streamlit as st
import time

st.set_page_config(
    page_title="Churn Guardian",
    layout="wide"
)

# Initialize session state for the welcome page
if 'welcome_complete' not in st.session_state:
    st.session_state.welcome_complete = False

def show_welcome_page():
    """Renders the Welcome page with animations and a button to proceed."""
    st.markdown("""<style> @keyframes fadeIn { 0% { opacity: 0; transform: translateY(20px); } 100% { opacity: 1; transform: translateY(0); } }
                    .animated-text { animation: fadeIn 2s ease-in-out; font-size: 3em; font-weight: bold; text-align: center; }
                    .subtitle-text { animation: fadeIn 3s ease-in-out; text-align: center; font-size: 1.5em; color: #a0a0a0; } </style>""",
                    unsafe_allow_html=True)
    
    st.markdown('<div class="animated-text">Welcome to Churn Guardian</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Your AI-Powered Retention Co-Pilot</div>', unsafe_allow_html=True)
    
    time.sleep(2) # Pause for animation

    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.write("---")
    st.subheader("Scroll down to begin")
    st.markdown("<br>" * 10, unsafe_allow_html=True)

    if st.button("Enter Application", use_container_width=True):
        st.session_state.welcome_complete = True
        st.switch_page("pages/1_Dashboard.py") # The key redirect command

# Main logic
if not st.session_state.welcome_complete:
    show_welcome_page()
else:
    # If the user is already "in", just switch them to the dashboard immediately.
    # This prevents them from being stuck on a blank welcome page if they navigate back.
    st.switch_page("pages/1_Dashboard.py")