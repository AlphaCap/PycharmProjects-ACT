import streamlit as st
from data_manager import get_trades_history_formatted, get_portfolio_metrics, get_positions
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Main Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit elements
hide_streamlit_style = """
    <style>
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
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Trading Systems")
    st.caption(f"Last Updated: {datetime.now().strftime('%m/%d/%Y %H:%M')}")
    st.markdown("---")
    if st.button("Go to nGS Performance", use_container_width=True):
        try:
            st.switch_page("pages/1_nGS_System.py")
        except streamlit.errors.StreamlitAPIException:
            st.warning("Failed to switch to nGS page. Ensure 'pages/1_nGS_System.py' exists.")

# Main content
st.title("Main Dashboard")

# Initialize session state for account size
if 'initial_value' not in st.session_state:
    st.session_state.initial_value = 1000000
initial_value = st.number_input("Set initial portfolio/account size:", min_value=1000, value=st.session_state.initial_value, step=1000, format="%d", key="account_size_input_home")
st.session_state.initial_value = initial_value

# Fetch portfolio metrics
metrics = get_portfolio_metrics(initial_value)
st.write("Portfolio Metrics:", metrics)

# Display Trades
trades = get_trades_history_formatted()
st.dataframe(trades)

# Display Positions
positions = get_positions()
if positions:
    st.write("Current Positions:", positions)
else:
    st.info("No current positions available.")

# System Status
st.markdown("---")
st.subheader("⚙️ System Status")
try:
    st.success("✅ System Online")
    st.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
except Exception as e:
    st.error(f"Error getting system status: {e}")

st.markdown("<p style='text-align: center; color: #999; font-size: 0.8rem;'>* Data retention: 6 months (180 days)</p>", unsafe_allow_html=True)
