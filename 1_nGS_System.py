import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# Set page config
st.set_page_config(page_title="nGS Trading System", layout="wide")

# Header
st.title("nGS Trading System")
st.caption("Long/Short Trading System based on EOD data")

# System metrics
st.markdown("## Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="YTD Return", value="18.2%", delta="7.2%")
with col2:
    st.metric(label="Sharpe Ratio", value="1.8", delta="0.2")
with col3:
    st.metric(label="Max Drawdown", value="-8.5%", delta="1.2%", delta_color="inverse")
with col4:
    st.metric(label="Win Rate", value="62%", delta="3%")

# Performance chart
st.markdown("## Performance History")

# Generate sample performance data
dates = pd.date_range(start='2025-01-01', end='2025-07-04')
cumulative_return = np.zeros(len(dates))
for i in range(1, len(dates)):
    # Random daily return between -2% and 2%
    daily_return = np.random.normal(0.06/252, 0.15/np.sqrt(252))
    cumulative_return[i] = cumulative_return[i-1] + daily_return

# Scale to match YTD return of 18.2%
final_return = 0.182
scale_factor = final_return / cumulative_return[-1]
cumulative_return = cumulative_return * scale_factor

# Create a DataFrame for the performance
performance_df = pd.DataFrame({
    'Date': dates,
    'nGS System': (1 + cumulative_return) * 100 - 100,
    'S&P 500': (1 + cumulative_return * 0.7) * 100 - 100  # Benchmark with lower return
})

# Plot performance chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(performance_df['Date'], performance_df['nGS System'], label='nGS System', linewidth=2)
ax.plot(performance_df['Date'], performance_df['S&P 500'], label='S&P 500', linewidth=2, linestyle='--')
ax.set_title('YTD Performance', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# Trade log section
st.markdown("## nGS Trade Log")

# Load trade log if available
csv_path = os.path.join("data", "trades.csv")
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Filter for nGS trades only (assuming 'system' column exists)
        if 'system' in df.columns:
            ngs_df = df[df['system'] == 'nGS'].copy()
            st.dataframe(ngs_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading trade log: {e}")
        
        # Example trade data when CSV not available
        trade_data = pd.DataFrame({
            "Symbol": ["AAPL", "META", "TSLA", "HD", "IBM", "INTC", "WMT"],
            "Entry Date": ["06/15/25", "06/25/25", "06/20/25", "07/01/25", "06/22/25", "06/15/25", "06/25/25"],
            "Side": ["Long", "Long", "Long", "Long", "Short", "Short", "Short"],
            "Entry Price": ["$190.25", "$490.30", "$240.50", "$345.20", "$185.30", "$45.70", "$68.40"],
            "Current Price": ["$205.50", "$510.60", "$255.40", "$352.80", "$178.80", "$42.40", "$66.20"],
            "Return": ["8.0%", "4.1%", "6.2%", "2.2%", "3.5%", "7.2%", "3.2%"],
            "Status": ["Open", "Open", "Open", "Open", "Open", "Open", "Open"]
        })
        
        st.dataframe(trade_data, use_container_width=True, hide_index=True)
else:
    # Example trade data when CSV not available
    trade_data = pd.DataFrame({
        "Symbol": ["AAPL", "META", "TSLA", "HD", "IBM", "INTC", "WMT"],
        "Entry Date": ["06/15/25", "06/25/25", "06/20/25", "07/01/25", "06/22/25", "06/15/25", "06/25/25"],
        "Side": ["Long", "Long", "Long", "Long", "Short", "Short", "Short"],
        "Entry Price": ["$190.25", "$490.30", "$240.50", "$345.20", "$185.30", "$45.70", "$68.40"],
        "Current Price": ["$205.50", "$510.60", "$255.40", "$352.80", "$178.80", "$42.40", "$66.20"],
        "Return": ["8.0%", "4.1%", "6.2%", "2.2%", "3.5%", "7.2%", "3.2%"],
        "Status": ["Open", "Open", "Open", "Open", "Open", "Open", "Open"]
    })
    
    st.dataframe(trade_data, use_container_width=True, hide_index=True)

# Strategy parameters section
with st.expander("Strategy Parameters"):
    st.markdown("""
    ### nGS System Parameters
    
    The nGS system is a long/short trading strategy that utilizes end-of-day data from S&P 500 component stocks.
    
    **Key Parameters:**
    - **Universe**: S&P 500 Components
    - **Time Frame**: Daily (End-of-Day)
    - **Signal Generation**: 4:00 PM ET
    - **Trade Execution**: Next day market open
    - **Position Sizing**: Equal weight with risk adjustment
    - **Max Long Exposure**: 100% of capital
    - **Max Short Exposure**: 50% of capital
    - **Stop Loss**: 8% individual position
    
    *For detailed methodology documentation, please contact the trading desk.*
    """)

# Go back to main dashboard button
if st.button("← Back to Dashboard"):
    st.switch_page("app.py")