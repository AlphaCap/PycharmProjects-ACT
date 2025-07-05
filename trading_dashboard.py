import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime

# Add path to ensure imports work
sys.path.append('C:/ACT/Python NEW 2025')

# Import minimal functionality to start
from main import VERSION

# Set page config with explicit path
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"  # Hardcoded API key
os.environ['POLYGON_API_KEY'] = API_KEY

# Main app
st.title("Trading Dashboard")
st.caption(f"nGS System Version {VERSION}")

# Simple interface
st.subheader("System Overview")

# Display basic info
col1, col2 = st.columns(2)
with col1:
    st.metric("Current Date", datetime.now().strftime("%Y-%m-%d"))
with col2:
    st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

# System info
st.text(f"Python Version: {sys.version.split()[0]}")
st.text(f"System Directory: {os.getcwd()}")
st.text(f"API Key: {API_KEY[:5]}...{API_KEY[-3:]}")

# Symbol search
st.divider()
st.subheader("Quick Symbol Search")
    
symbol = st.text_input("Enter Symbol (e.g., AAPL)", max_chars=5).upper()

if st.button("Search"):
    if symbol:
        st.info(f"You entered: {symbol}")
        st.text("Processing symbol...")
    else:
        st.warning("Please enter a symbol")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
