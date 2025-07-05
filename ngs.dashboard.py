import streamlit as st
import os
import sys
from datetime import datetime

# Add path to ensure imports work
sys.path.append('C:/ACT/Python NEW 2025')

# Import only what you need from nGS, avoiding the other systems
from main import VERSION
# Avoid importing modules related to Alpha Capture AI and gST DayTrader

st.set_page_config(
    page_title="nGS Trading System",  # Use very specific name
    page_icon="📈",
    layout="wide"
)

# Main app
st.title("nGS Trading System Dashboard")
st.caption(f"Version {VERSION} - Neural Grid Strategy Only")

# Clarify that other systems are inactive
st.info("⚠️ Alpha Capture AI and gST DayTrader are currently inactive")

# Display basic info
col1, col2 = st.columns(2)
with col1:
    st.metric("Current Date", datetime.now().strftime("%Y-%m-%d"))
with col2:
    st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

# Add nGS-specific functionality here
# ...

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
