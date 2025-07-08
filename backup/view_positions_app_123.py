import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# Force a unique page path with very explicit configuration
st.set_page_config(
    page_title="Neural Grid Strategy Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Neural Grid Strategy Dashboard\nVersion 1.0"
    }
)

# Custom title with no potential conflicts
st.title("📊 Neural Grid Strategy Position Viewer")
st.caption("Isolated App - No Module Imports")

# Display current time
col1, col2 = st.columns(2)
with col1:
    st.metric("Current Date", datetime.now().strftime("%Y-%m-%d"))
with col2:
    st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

# Read positions directly from file
positions_data = []
try:
    if os.path.exists("data/positions.json"):
        with open("data/positions.json", "r") as f:
            positions = json.load(f)
            
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
        else:
            st.info("No position data found in file.")
    else:
        st.warning("No positions data file found. Run export_data.py first.")
except Exception as e:
    st.error(f"Error loading positions: {str(e)}")

# Footer
st.divider()
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("Refresh Data"):
    st.experimental_rerun()