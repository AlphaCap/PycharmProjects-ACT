import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import json

# Set page config - this DOESN'T cause conflicts
st.set_page_config(
    page_title="nGS Trading System",
    page_icon="📈",
    layout="wide"
)

# Read VERSION directly from main.py without importing it
VERSION = "1.0.0"  # Default value
try:
    with open("main.py", "r") as f:
        for line in f:
            if line.startswith("VERSION ="):
                VERSION = line.split("=")[1].strip().strip('"\'')
                break
except:
    pass

# Function to get positions without importing modules
def get_positions():
    try:
        # Try to read directly from your data storage
        if os.path.exists("data/positions.json"):
            with open("data/positions.json", "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading positions: {str(e)}")
        return {}

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

# Get LIVE position data
positions = get_positions()

# Show positions
st.subheader("Current Positions")

if positions:
    # Convert positions to dataframe for display
    positions_data = []
    for symbol, pos in positions.items():
        positions_data.append({
            "Symbol": symbol,
            "Shares": pos.get("shares", 0),
            "Entry Price": f"${pos.get('entry_price', 0):.2f}",
            "Entry Date": pos.get("entry_date", ""),
            "Days Held": pos.get("bars_since_entry", 0),
            "P&L": f"${pos.get('current_profit', 0):.2f}"
        })
    
    if positions_data:
        st.dataframe(pd.DataFrame(positions_data))
    else:
        st.info("No active positions found.")
else:
    st.info("No position data available. Check data storage.")
    
    # Add a data export function button
    if st.button("Export Positions Data"):
        # This code will create the necessary data files
        st.info("Creating data export script...")
        export_code = """
import json
from data_manager import get_positions

# Get position data
positions = get_positions()

# Export to JSON file
with open("data/positions.json", "w") as f:
    json.dump(positions, f, indent=4)

print("Position data exported to data/positions.json")
        """
        st.code(export_code, language="python")
        st.info("Run the script above in your Python environment to export position data")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
