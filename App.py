import streamlit as st
import pandas as pd
import os
import datetime

# Set page config for wide layout
st.set_page_config(
    page_title="Alpha Trading Systems Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
with st.sidebar:
    st.title("Trading Systems")
    
    # System navigation buttons
    st.subheader("Trading Performance")
    if st.button("nGS System", use_container_width=True):
        st.switch_page("pages/1_nGS_System.py")
    
    if st.button("Daily System", use_container_width=True):
        st.switch_page("pages/2_Daily_System.py")
    
    if st.button("Intraday System (Polygon)", use_container_width=True):
        st.switch_page("pages/3_Intraday_System.py")
    
    # Additional sidebar info
    st.markdown("---")
    st.caption("Data last updated:")
    st.caption(f"{datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}")

# Page header with branding
st.title("Alpha Trading Systems Dashboard")
st.caption("S&P 500 Component Trading")

# Summary metrics
st.markdown("## Current Portfolio Status")

# Create tabs for summary view
tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Long Positions", "Short Positions"])

with tab1:
    # Portfolio overview section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Portfolio Value", value="$1,250,000", delta="2.5%")
    with col2:
        st.metric(label="Daily P&L", value="$12,500", delta="1.0%")
    with col3:
        st.metric(label="MTD Return", value="4.2%", delta="0.8%")
    with col4:
        st.metric(label="YTD Return", value="15.6%", delta="7.2%")
    
    # Strategy allocation and performance table
    st.subheader("Strategy Performance")
    
    # Load portfolio data if available
    try:
        # Placeholder for portfolio data - replace with actual data loading
        portfolio_data = pd.DataFrame({
            "Strategy": ["nGS", "Daily System", "Intraday System"],
            "Allocation": ["$600,000", "$400,000", "$250,000"],
            "Daily Return": ["1.2%", "0.8%", "0.5%"],
            "MTD Return": ["5.1%", "3.8%", "2.9%"],
            "YTD Return": ["18.2%", "14.6%", "11.2%"],
            "Sharpe": ["1.8", "1.6", "1.4"]
        })
        
        st.dataframe(portfolio_data, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading portfolio data: {e}")

    # Top performing positions
    st.subheader("Top Performing Positions")
    
    # Example top positions data
    top_positions = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"],
        "Strategy": ["nGS", "Intraday", "nGS", "Daily", "Intraday"],
        "Entry Date": ["06/15/25", "07/01/25", "06/22/25", "06/28/25", "07/02/25"],
        "Entry Price": ["$190.25", "$415.80", "$178.60", "$175.40", "$120.75"],
        "Current Price": ["$205.50", "$440.20", "$188.30", "$182.60", "$126.40"],
        "Return": ["8.0%", "5.9%", "5.4%", "4.1%", "4.7%"],
        "Side": ["Long", "Long", "Long", "Short", "Long"]
    })
    
    st.dataframe(top_positions, use_container_width=True, hide_index=True)

with tab2:
    # Long positions section
    st.subheader("Active Long Positions")
    
    # Load long positions data if available
    try:
        # Path to your CSV (replace with actual data loading)
        csv_path = os.path.join("data", "trades.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter for long positions only (assuming 'side' column contains this info)
            long_df = df[df['side'].str.lower().str.startswith('b')].copy() if 'side' in df.columns else pd.DataFrame()
            
            if not long_df.empty:
                # Display long positions
                st.dataframe(long_df, use_container_width=True, hide_index=True)
            else:
                st.info("No active long positions found.")
        else:
            # Example long positions data when CSV not available
            long_positions = pd.DataFrame({
                "Symbol": ["AAPL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "V", "HD", "PG", "UNH"],
                "Strategy": ["nGS", "Daily", "Intraday", "nGS", "Intraday", "nGS", "Daily", "nGS", "Intraday", "Daily"],
                "Entry Date": ["06/15/25", "07/01/25", "06/22/25", "06/25/25", "07/02/25", "06/20/25", "06/28/25", "07/01/25", "06/27/25", "06/15/25"],
                "Entry Price": ["$190.25", "$415.80", "$178.60", "$490.30", "$120.75", "$240.50", "$275.60", "$345.20", "$165.70", "$540.30"],
                "Current Price": ["$205.50", "$440.20", "$188.30", "$510.60", "$126.40", "$255.40", "$282.10", "$352.80", "$172.40", "$556.20"],
                "Return": ["8.0%", "5.9%", "5.4%", "4.1%", "4.7%", "6.2%", "2.4%", "2.2%", "4.0%", "2.9%"],
                "Status": ["Open", "Open", "Open", "Open", "Open", "Open", "Open", "Open", "Open", "Open"]
            })
            
            st.dataframe(long_positions, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading long positions: {e}")

with tab3:
    # Short positions section
    st.subheader("Active Short Positions")
    
    # Load short positions data if available
    try:
        # Path to your CSV (replace with actual data loading)
        csv_path = os.path.join("data", "trades.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter for short positions only
            short_df = df[df['side'].str.lower().str.startswith('s')].copy() if 'side' in df.columns else pd.DataFrame()
            
            if not short_df.empty:
                # Display short positions
                st.dataframe(short_df, use_container_width=True, hide_index=True)
            else:
                st.info("No active short positions found.")
        else:
            # Example short positions data when CSV not available
            short_positions = pd.DataFrame({
                "Symbol": ["GOOGL", "IBM", "INTC", "DIS", "KO", "WMT", "JPM", "BA"],
                "Strategy": ["Daily", "nGS", "nGS", "Intraday", "Daily", "nGS", "Intraday", "Daily"],
                "Entry Date": ["06/28/25", "06/22/25", "06/15/25", "06/30/25", "07/01/25", "06/25/25", "06/28/25", "07/02/25"],
                "Entry Price": ["$175.40", "$185.30", "$45.70", "$98.60", "$62.80", "$68.40", "$182.30", "$190.50"],
                "Current Price": ["$168.20", "$178.80", "$42.40", "$95.30", "$60.90", "$66.20", "$175.60", "$184.80"],
                "Return": ["4.1%", "3.5%", "7.2%", "3.3%", "3.0%", "3.2%", "3.7%", "3.0%"],
                "Status": ["Open", "Open", "Open", "Open", "Open", "Open", "Open", "Open"]
            })
            
            st.dataframe(short_positions, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading short positions: {e}")

# Recent activity section
st.markdown("## Recent Activity")

# Create two columns for recent signals and actions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Signals")
    
    # Example recent signals data
    signals = [
        {"date": "07/04/25", "symbol": "AAPL", "signal": "L", "strategy": "nGS", "price": "$205.50"},
        {"date": "07/03/25", "symbol": "MSFT", "signal": "L", "strategy": "Intraday", "price": "$440.20"},
        {"date": "07/03/25", "symbol": "IBM", "signal": "S", "strategy": "nGS", "price": "$178.80"},
        {"date": "07/02/25", "symbol": "NVDA", "signal": "L", "strategy": "Intraday", "price": "$126.40"},
        {"date": "07/02/25", "symbol": "BA", "signal": "S", "strategy": "Daily", "price": "$184.80"}
    ]
    
    for signal in signals:
        st.markdown(f"- **{signal['date']}** | **{signal['symbol']}** | **{signal['signal']}** | {signal['strategy']} at `{signal['price']}`")

with col2:
    st.subheader("System Status")
    
    # Example system status data
    system_status = [
        {"timestamp": "07/04/25 16:30", "system": "nGS", "message": "Daily signals generated successfully"},
        {"timestamp": "07/04/25 16:15", "system": "Intraday", "message": "Trading session closed, 8 signals processed"},
        {"timestamp": "07/04/25 15:45", "system": "Daily", "message": "Portfolio rebalanced, 2 new positions"},
        {"timestamp": "07/04/25 09:30", "system": "Intraday", "message": "Trading session started"},
        {"timestamp": "07/03/25 16:30", "system": "nGS", "message": "Daily signals generated successfully"}
    ]
    
    for status in system_status:
        st.markdown(f"- **{status['timestamp']}** | **{status['system']}** | {status['message']}")

# Footer with additional info
st.markdown("---")
st.caption("Alpha Trading Systems Dashboard - For additional support, please contact the trading desk.")
