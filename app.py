import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import (
    get_portfolio_metrics,
    get_signals,
    get_positions,
    get_system_status
)

try:
    from portfolio_calculator import calculate_real_portfolio_metrics
    USE_REAL_METRICS = True
except ImportError:
    USE_REAL_METRICS = False

st.set_page_config(
    page_title="nGS Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit style elements
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

with st.sidebar:
    st.title("nGS Trading System")
    
    # Navigation
    if st.button("📊 Historical Performance", use_container_width=True):
        st.switch_page("pages/1_nGS_System.py")
    
    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# Main content
st.title("🎯 nGS Trading Dashboard")

# Initialize session state for account size
if 'account_size' not in st.session_state:
    st.session_state.account_size = 1000000

# Account size input
account_size = st.number_input(
    "Account Size:",
    min_value=1000,
    value=st.session_state.account_size,
    step=1000,
    format="%d",
    key="main_account_size"
)
st.session_state.account_size = account_size

# Get portfolio metrics with fallback
def get_portfolio_metrics_with_fallback(account_size: int) -> dict:
    try:
        if USE_REAL_METRICS:
            return calculate_real_portfolio_metrics(initial_portfolio_value=account_size)
        return get_portfolio_metrics(initial_portfolio_value=account_size)
    except Exception as e:
        st.error(f"Error getting portfolio metrics: {e}")
        return {
            'total_value': f"${account_size:,.0f}",
            'total_return_pct': "+0.0%",
            'daily_pnl': "$0.00",
            'unrealized_pnl': "$0.00",
            'realized_pnl': "$0.00",
            'win_rate': "0.0%"
        }

# Load metrics
metrics = get_portfolio_metrics_with_fallback(account_size)

# Ensure all required keys exist
safe_metrics = {
    'total_value': f"${account_size:,.0f}",
    'total_return_pct': "+0.0%",
    'daily_pnl': "$0.00",
    'unrealized_pnl': "$0.00",
    'realized_pnl': "$0.00",
    'win_rate': "0.0%"
}
for key, default_value in safe_metrics.items():
    if key not in metrics:
        metrics[key] = default_value

# Portfolio Summary
st.subheader("💰 Portfolio Summary")

# Custom CSS for smaller metric fonts
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.1);
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
    font-size: 0.8rem;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_value_clean = str(metrics['total_value']).replace('.00', '').replace(',', '')
    st.metric(label="Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])

with col2:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])

with col3:
    st.metric(label="Unrealized P&L", value=metrics['unrealized_pnl'])

with col4:
    st.metric(label="Realized P&L", value=metrics['realized_pnl'])

with col5:
    st.metric(label="Win Rate", value=metrics['win_rate'])

with col6:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Current Positions
st.subheader("📋 Current Positions")

try:
    positions = get_positions()
    
    if positions and len(positions) > 0:
        # Convert to DataFrame for better display
        df_positions = pd.DataFrame(positions)
        
        # Ensure required columns exist
        required_columns = ['symbol', 'entry_date', 'side', 'shares', 'entry_price', 'current_price', 'profit']
        for col in required_columns:
            if col not in df_positions.columns:
                if col == 'side':
                    df_positions[col] = 'long'  # default
                elif col == 'entry_date':
                    df_positions[col] = datetime.now().strftime('%Y-%m-%d')  # default to today
                elif col in ['shares', 'entry_price', 'current_price']:
                    df_positions[col] = 0
                elif col == 'profit':
                    df_positions[col] = 0.0
                else:
                    df_positions[col] = 'N/A'
        
        # Format the display DataFrame with entry_date as 2nd column
        display_df = pd.DataFrame()
        display_df['Symbol'] = df_positions['symbol']
        display_df['Entry Date'] = pd.to_datetime(df_positions['entry_date']).dt.strftime('%m/%d/%y')
        display_df['Side'] = df_positions['side'].str.capitalize()
        display_df['Shares'] = df_positions['shares'].astype(int)
        display_df['Entry Price'] = df_positions['entry_price'].apply(lambda x: f"${float(x):,.2f}")
        display_df['Current Price'] = df_positions['current_price'].apply(lambda x: f"${float(x):,.2f}")
        display_df['Current Value'] = (df_positions['shares'] * df_positions['current_price']).apply(lambda x: f"${float(x):,.0f}")
        display_df['P&L'] = df_positions['profit'].apply(lambda x: f"${float(x):+,.0f}")
        display_df['P&L %'] = (
            (df_positions['current_price'] / df_positions['entry_price'] - 1) * 100 * 
            df_positions['shares'].apply(lambda x: 1 if x > 0 else -1)
        ).apply(lambda x: f"{float(x):+.1f}%")
        
        # Display positions count
        total_positions = len(display_df)
        long_count = len(df_positions[df_positions['side'] == 'long'])
        short_count = len(df_positions[df_positions['side'] == 'short'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Positions", total_positions)
        with col2:
            st.metric("Long Positions", long_count)
        with col3:
            st.metric("Short Positions", short_count)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Entry Date": st.column_config.TextColumn("Entry Date", width="small"),
                "Side": st.column_config.TextColumn("Side", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", width="small"),
                "Entry Price": st.column_config.TextColumn("Entry Price", width="medium"),
                "Current Price": st.column_config.TextColumn("Current Price", width="medium"),
                "Current Value": st.column_config.TextColumn("Current Value", width="medium"),
                "P&L": st.column_config.TextColumn("P&L", width="medium"),
                "P&L %": st.column_config.TextColumn("P&L %", width="small")
            }
        )
        
        # Check for suspicious future dates
        future_positions = df_positions[pd.to_datetime(df_positions['entry_date']) > datetime.now()]
        if not future_positions.empty:
            st.warning(f"⚠️ WARNING: {len(future_positions)} positions have FUTURE entry dates - these are likely from backtest data!")
            st.info("🔍 If you see future dates (like 04/09/25), these positions are synthetic/test data, not real trades.")
        
        # Check for very recent positions (today)
        today = datetime.now().strftime('%Y-%m-%d')
        today_positions = df_positions[df_positions['entry_date'].str.contains(today, na=False)]
        if not today_positions.empty:
            st.success(f"✅ {len(today_positions)} positions entered TODAY - these appear to be new live trades!")
    
    else:
        st.info("No current positions")

except Exception as e:
    st.error(f"Error loading positions: {e}")
    st.info("Check your data sources and ensure the system is properly configured.")

st.markdown("---")

# Today's Signals
st.subheader("🎯 Today's Signals")
try:
    signals = get_signals()
    
    # Check if signals exist and are not empty
    if signals is not None:
        if isinstance(signals, pd.DataFrame):
            if not signals.empty:
                # Show last 10 signals for DataFrame
                recent_signals = signals.head(10)
                st.dataframe(recent_signals, use_container_width=True, hide_index=True)
            else:
                st.info("No signals today")
        elif isinstance(signals, list) and len(signals) > 0:
            # Show last 10 signals for list
            recent_signals = signals[:10]
            signal_df = pd.DataFrame(recent_signals)
            st.dataframe(signal_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals today")
    else:
        st.info("No signals today")
        
except Exception as e:
    st.error(f"Error loading signals: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    nGulfStream Trading System | Neural Grid Strategy<br>
    <em>Real-time algorithmic trading dashboard</em>
    </div>
    """, 
    unsafe_allow_html=True
)
