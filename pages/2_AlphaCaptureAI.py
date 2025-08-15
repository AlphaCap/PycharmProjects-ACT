import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(page_title="Daily Trading System", layout="wide")

# Header
st.title("Daily Trading System")
st.caption("Placeholder for Daily EOD Trading System")

st.markdown(
    """
## Coming Soon

This trading system is currently under development. It will utilize EOD data from S&P 500 component stocks.

Check back for updates.
"""
)

# Go back to main dashboard button
if st.button("← Back to Dashboard"):
    st.switch_page("ngs_ai_performance_comparator")
