# Hide Streamlit style elements and add professional background
hide_streamlit_style = """
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stHeader"] {display: none;}
    .stApp > header {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    
    .stAppViewContainer > .main .block-container {
        padding-top: 1rem;
    }
    
    /* Professional trading dashboard background */
    .stApp {
        background: linear-gradient(135deg, 
            #0f1419 0%, 
            #1a202c 25%, 
            #2d3748 50%, 
            #1a202c 75%, 
            #0f1419 100%);
        background-attachment: fixed;
    }
    
    /* Subtle overlay pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(120, 119, 198, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.01) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 119, 198, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Enhance metric containers for better contrast */
    [data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.15);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhance main content containers */
    .stApp > div > div > div > div {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Sidebar styling to match */
    .css-1d391kg {
        background: linear-gradient(180deg, 
            rgba(15, 20, 25, 0.95) 0%, 
            rgba(26, 32, 44, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    </style>
"""
