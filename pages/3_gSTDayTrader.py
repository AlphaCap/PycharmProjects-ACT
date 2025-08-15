import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Import API key from config
from config import POLYGON_API_KEY

# Add the utils directory to the path to import polygon_api
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from utils.polygon_api import PolygonClient
except ImportError:
    st.error(
        "Could not import PolygonClient. Make sure utils/polygon_api.py is available."
    )

# Set page config
st.set_page_config(page_title="Intraday Trading System", layout="wide")

# Header
st.title("Intraday Trading System (Polygon.io)")
st.caption("1-Minute Bar Trading Strategy")

# Use API key from config instead of user input
os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY

st.markdown("## Polygon.io API Status")
st.success("API Key loaded from config.py")

if st.button("Test Connection"):
    try:
        # Create client instance
        client = PolygonClient(POLYGON_API_KEY)

        # Test with a simple API call
        test_symbol = "AAPL"
        from_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime(
            "%Y-%m-%d"
        )
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")

        with st.spinner(f"Testing API connection with {test_symbol} data..."):
            test_data = client.get_bars(
                test_symbol, from_date=from_date, to_date=to_date
            )

            if not test_data.empty:
                st.success(
                    f"Connection successful! Retrieved {len(test_data)} bars for {test_symbol}."
                )

                # Show sample of the data
                st.subheader("Sample Data")
                st.dataframe(test_data.head(5))
            else:
                st.warning(
                    "Connection successful but no data returned. Try a different date range."
                )
    except Exception as e:
        st.error(f"Connection failed: {e}")

# System placeholder
st.markdown("## Intraday System")
st.markdown(
    "This system will utilize 1-minute bar data from Polygon.io to generate trading signals."
)

# Placeholder for future functionality
st.markdown(
    """
### Coming Soon:

- Real-time signal generation
- Intraday performance metrics
- Position management
- Historical backtesting tools

Once API setup is complete, this page will display the Intraday Trading System dashboard.
"""
)

# Go back to main dashboard button
if st.button("← Back to Dashboard"):
    st.switch_page("app.py")
