import streamlit as st
import os
import base64

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Churn Guardian",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------
# HELPER FUNCTION TO ENCODE VIDEO
# --------------------
def get_video_as_base64(path):
    """Reads a video file and returns its Base64 encoded string."""
    if not os.path.exists(path):
        st.error(f"Video not found at path: {path}")
        return None
    
    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode()

# --------------------
# SESSION STATE INITIALIZATION
# --------------------
if "welcome_complete" not in st.session_state:
    st.session_state.welcome_complete = False

# --------------------
# WELCOME PAGE FUNCTION
# --------------------
def show_welcome_page():
    """Displays the fullscreen, interactive, scroll-based welcome page with video background."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "oo.mp4")
    
    video_base64 = get_video_as_base64(video_path)

    # Create video HTML if video exists
    video_html = ""
    if video_base64:
        video_html = f'<video autoplay muted loop playsinline><source src="data:video/mp4;base64,{video_base64}" type="video/mp4"></video>'

    st.markdown(
        f"""
        <style>
            /* --- General Page Styling --- */
            #MainMenu, footer, header {{ visibility: hidden; }}
            
            /* Make Streamlit containers transparent */
            .main .block-container {{ 
                padding: 0;
                background-color: transparent !important;
            }}
            .main {{
                background-color: transparent !important;
            }}
            .stApp {{
                background-color: transparent !important;
            }}
            
            /* Smooth scrolling */
            html {{
                scroll-behavior: smooth;
            }}
            
            /* --- Background Video --- */
            .bg-video {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -2;
                overflow: hidden;
            }}
            
            .bg-video video {{
                position: absolute;
                top: 50%;
                left: 50%;
                min-width: 100%;
                min-height: 100%;
                width: auto;
                height: auto;
                transform: translate(-50%, -50%);
                object-fit: cover;
                filter: brightness(0.5);
            }}
            
            /* --- Animated Grid Overlay --- */
            .grid-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-image: 
                    linear-gradient(rgba(0, 179, 255, 0.1) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 179, 255, 0.1) 1px, transparent 1px);
                background-size: 50px 50px;
                z-index: -1;
                animation: gridMove 20s linear infinite;
            }}
            
            /* --- Floating Particles --- */
            .particle {{
                position: fixed;
                width: 4px;
                height: 4px;
                background: #00ffff;
                border-radius: 50%;
                box-shadow: 0 0 10px #00ffff;
                z-index: -1;
                pointer-events: none;
            }}
            
            .particle:nth-child(1) {{ left: 10%; animation: float 8s ease-in-out infinite; animation-delay: 0s; }}
            .particle:nth-child(2) {{ left: 20%; animation: float 10s ease-in-out infinite; animation-delay: 1s; }}
            .particle:nth-child(3) {{ left: 30%; animation: float 7s ease-in-out infinite; animation-delay: 2s; }}
            .particle:nth-child(4) {{ left: 40%; animation: float 9s ease-in-out infinite; animation-delay: 1.5s; }}
            .particle:nth-child(5) {{ left: 50%; animation: float 11s ease-in-out infinite; animation-delay: 0.5s; }}
            .particle:nth-child(6) {{ left: 60%; animation: float 8s ease-in-out infinite; animation-delay: 2.5s; }}
            .particle:nth-child(7) {{ left: 70%; animation: float 10s ease-in-out infinite; animation-delay: 1s; }}
            .particle:nth-child(8) {{ left: 80%; animation: float 9s ease-in-out infinite; animation-delay: 3s; }}
            .particle:nth-child(9) {{ left: 90%; animation: float 7s ease-in-out infinite; animation-delay: 0.8s; }}
            
            /* --- Scan Lines Effect --- */
            .scanlines {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: linear-gradient(
                    transparent 50%,
                    rgba(0, 255, 255, 0.03) 50%
                );
                background-size: 100% 4px;
                z-index: -1;
                pointer-events: none;
                animation: scan 8s linear infinite;
            }}
            
            /* --- Corner Brackets --- */
            .corner-brackets {{
                position: fixed;
                width: 100px;
                height: 100px;
                border: 2px solid #00ffff;
                z-index: 1;
                pointer-events: none;
            }}
            
            .corner-brackets.top-left {{
                top: 20px;
                left: 20px;
                border-right: none;
                border-bottom: none;
                animation: pulse 3s ease-in-out infinite;
            }}
            
            .corner-brackets.top-right {{
                top: 20px;
                right: 20px;
                border-left: none;
                border-bottom: none;
                animation: pulse 3s ease-in-out infinite 0.5s;
            }}
            
            .corner-brackets.bottom-left {{
                bottom: 20px;
                left: 20px;
                border-right: none;
                border-top: none;
                animation: pulse 3s ease-in-out infinite 1s;
            }}
            
            .corner-brackets.bottom-right {{
                bottom: 20px;
                right: 20px;
                border-left: none;
                border-top: none;
                animation: pulse 3s ease-in-out infinite 1.5s;
            }}
            
            /* --- Scrollable Sections --- */
            .scroll-section {{
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                position: relative;
            }}

            /* --- Animated Text & Glow Effect --- */
            .title-text {{
                font-size: 4.5rem;
                font-weight: bold;
                color: #ffffff;
                text-shadow: 
                    0 0 10px #00b3ff, 0 0 20px #00b3ff,
                    0 0 40px #00b3ff, 0 0 80px #00b3ff;
                animation: slideUp 1.5s cubic-bezier(0.19, 1, 0.22, 1), 
                           glow 2.5s ease-in-out infinite alternate;
                letter-spacing: 3px;
            }}
            
            .subtitle-text {{
                font-size: 1.8rem;
                color: rgba(255, 255, 255, 0.85);
                margin-top: 1rem;
                animation: slideUp 1.5s cubic-bezier(0.19, 1, 0.22, 1) 0.3s;
                animation-fill-mode: backwards;
                letter-spacing: 2px;
            }}
            
            /* --- Description Text --- */
            .description-text {{
                font-size: 1.1rem;
                color: rgba(255, 255, 255, 0.75);
                margin-top: 2rem;
                max-width: 800px;
                line-height: 1.6;
                animation: fadeIn 2s ease-in-out 0.6s;
                animation-fill-mode: backwards;
            }}
            
            /* --- Enter Button Section --- */
            .button-section-content {{
                 animation: fadeIn 2s ease-in-out;
            }}
            
            .button-section-content h2 {{
                font-size: 2.5rem;
                color: #fff;
                margin-bottom: 1rem;
                letter-spacing: 2px;
            }}
            
            .button-section-content p {{
                font-size: 1.3rem;
                color: rgba(255, 255, 255, 0.8);
                margin-bottom: 2rem;
                max-width: 700px;
            }}

            /* Style Streamlit's button */
            .stButton > button {{
                font-size: 1.5rem;
                font-weight: bold;
                color: #00ffff;
                padding: 1rem 3rem;
                border: 2px solid #00ffff;
                border-radius: 50px;
                background-color: rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease-in-out;
                letter-spacing: 2px;
                text-transform: uppercase;
            }}
            .stButton > button:hover {{
                background-color: #00ffff;
                color: #000000;
                border-color: #00ffff;
                box-shadow: 0 0 30px #00ffff, 0 0 50px #00ffff;
                transform: scale(1.05);
            }}

            /* --- Scroll Down Indicator --- */
            .scroll-indicator {{
                position: absolute;
                bottom: 30px;
                left: 50%;
                transform: translateX(-50%);
                color: white;
                font-size: 1rem;
                opacity: 0.7;
                animation: bounce 2s infinite, fadeIn 3s ease-in-out;
                letter-spacing: 2px;
                text-transform: uppercase;
            }}
            
            .scroll-arrow {{
                font-size: 2rem;
                display: block;
                margin-top: 0.5rem;
            }}

            /* --- Keyframe Animations --- */
            @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(100px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            @keyframes glow {{
                from {{ text-shadow: 0 0 10px #00b3ff, 0 0 20px #00b3ff; }}
                to {{ text-shadow: 0 0 20px #00b3ff, 0 0 40px #00b3ff, 0 0 60px #00b3ff; }}
            }}
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translate(-50%, 0); }}
                40% {{ transform: translate(-50%, -15px); }}
                60% {{ transform: translate(-50%, -7px); }}
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.3; transform: scale(1); }}
                50% {{ opacity: 1; transform: scale(1.05); }}
            }}
            @keyframes float {{
                0%, 100% {{ transform: translateY(0) translateX(0); opacity: 0; }}
                10% {{ opacity: 1; }}
                90% {{ opacity: 1; }}
                50% {{ transform: translateY(-100vh) translateX(50px); }}
            }}
            @keyframes gridMove {{
                0% {{ transform: translateY(0); }}
                100% {{ transform: translateY(50px); }}
            }}
            @keyframes scan {{
                0% {{ transform: translateY(-100%); }}
                100% {{ transform: translateY(100%); }}
            }}
        </style>

        <div class="bg-video">
            {video_html}
        </div>
        
        <div class="grid-overlay"></div>
        <div class="scanlines"></div>
        
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        
        <div class="corner-brackets top-left"></div>
        <div class="corner-brackets top-right"></div>
        <div class="corner-brackets bottom-left"></div>
        <div class="corner-brackets bottom-right"></div>
        
        <div class="scroll-section">
            <div class="title-text">CHURN GUARDIAN CO.</div>
            <div class="subtitle-text">The best ML-powered retention Copilot</div>
            <div class="description-text">
                Welcome to our Customer Churn Insights Platform. 
                This is a data-driven tool that predicts which customers are likely to discontinue their insurance policies. 
                Using explainable techniques like SHAP and LIME, we uncover the key factors behind every prediction, 
                helping companies make smarter, more targeted retention decisions.
            </div>
            <div class="scroll-indicator">
                Discover
                <span class="scroll-arrow">↓</span>
            </div>
        </div>

        <div class="scroll-section">
            <div class="button-section-content">
                <h2>Predict. Understand. Retain.</h2>
                <p>Insights that explain why customers churn, not just who will.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Enter", key="enter_button", use_container_width=True):
            st.session_state.welcome_complete = True
            st.switch_page("pages/1_Dashboard.py")

# --------------------
# MAIN APP FUNCTION
# --------------------
def show_main_app():
    """Displays the main dashboard content after the welcome page."""
    st.sidebar.title("Navigation")
    st.sidebar.page_link("app.py", label="Home")

    st.title("Main Dashboard")
    st.write("Welcome to the main application!")

# --------------------
# ROUTING LOGIC
# --------------------
if not st.session_state.welcome_complete:
    show_welcome_page()
else:
    
    st.switch_page("pages/1_Dashboard.py")
