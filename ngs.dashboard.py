import streamlit as st
import os
from datetime import datetime
import pandas as pd
import json

# IMPORTANT: Remove the path addition that's causing conflicts
# sys.path.append('C:/ACT/Python NEW 2025')

# IMPORTANT: Hardcode VERSION instead of importing it
VERSION = "1.0.0"  # Hardcoded version

st.set_page_config(
    page_title="nGS Trading System View",  # Changed to make it unique
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

# Read positions data from file
try:
    if os.path.exists("data/positions.json"):
        with open("data/positions.json", "r") as f:
            positions = json.load(f)
        
        # Create dataframe from positions
        positions_data = []
        for symbol, pos in positions.items():
            positions_data.append({
                "Symbol": symbol,
                "Shares": pos.get("shares", 0),
                "Entry Price": f"${pos.get('entry_price', 0):.2f}",
                "Entry Date": pos.get("entry_date", ""),
                "Days Held": pos.get("bars_since_entry", 0)
            })
        
        if positions_data:
            st.subheader("Current Positions")
            st.dataframe(pd.DataFrame(positions_data))
except Exception as e:
    st.error(f"Error loading positions: {str(e)}")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
