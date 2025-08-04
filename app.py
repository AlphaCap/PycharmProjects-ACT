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
st.title("nGulfStream Trader")

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

# Calculate L/S Ratio from current positions - CORRECTED FORMAT
def calculate_ls_ratio():
    try:
        positions = get_positions()
        if not positions or len(positions) == 0:
            return "N/A"
        
        df_positions = pd.DataFrame(positions)
        
        # Ensure required columns exist
        if 'side' not in df_positions.columns:
            # Determine side from shares if side column missing
            if 'shares' in df_positions.columns:
                df_positions['side'] = df_positions['shares'].apply(lambda x: 'long' if x > 0 else 'short')
            else:
                return "N/A"
        
        # Calculate position values (not just counts)
        if all(col in df_positions.columns for col in ['current_price', 'shares']):
            df_positions['position_value'] = df_positions['current_price'] * df_positions['shares'].abs()
            
            long_value = df_positions[df_positions['side'] == 'long']['position_value'].sum()
            short_value = df_positions[df_positions['side'] == 'short']['position_value'].sum()
            
            if long_value == 0 and short_value == 0:
                return "N/A"
            elif short_value == 0:
                return f"{long_value/short_value if short_value > 0 else int(long_value/1000)}:0"  # All long
            elif long_value == 0:
                return f"-{int(short_value/1000)}:0"  # All short with minus
            else:
                # Calculate the ratio based on VALUES
                if long_value >= short_value:
                    ratio = long_value / short_value
                    return f"{ratio:.1f}:1"  # Positive for net long
                else:
                    ratio = short_value / long_value
                    return f"-{ratio:.1f}:1"  # NEGATIVE for net short (no letter S)
        else:
            # Fallback to position counts if values not available
            long_count = len(df_positions[df_positions['side'] == 'long'])
            short_count = len(df_positions[df_positions['side'] == 'short'])
            
            if long_count == 0 and short_count == 0:
                return "N/A"
            elif short_count == 0:
                return f"{long_count}:0"
            elif long_count == 0:
                return f"-{short_count}:0"  # Negative for shorts only
            else:
                if long_count >= short_count:
                    ratio = long_count / short_count
                    return f"{ratio:.1f}:1"
                else:
                    ratio = short_count / long_count
                    return f"-{ratio:.1f}:1"  # Negative for net short
                    
    except Exception as e:
        return f"Error"

# Enhanced portfolio metrics calculation - CORRECTED
def get_enhanced_portfolio_metrics(account_size: int) -> dict:
    try:
        # Get base metrics
        if USE_REAL_METRICS:
            metrics = calculate_real_portfolio_metrics(initial_portfolio_value=account_size)
        else:
            metrics = get_portfolio_metrics(initial_portfolio_value=account_size)
        
        total_realized = 0.0
        total_unrealized = 0.0
        
        # Get realized P&L from trades data
        try:
            import os
            if os.path.exists('data/trades.csv'):
                trades_df = pd.read_csv('data/trades.csv')
                if 'profit' in trades_df.columns and len(trades_df) > 0:
                    total_realized = trades_df['profit'].sum()
                    metrics['realized_pnl'] = f"${total_realized:+,.0f}"
            elif os.path.exists('trade_history.csv'):
                trades_df = pd.read_csv('trade_history.csv')
                if 'profit' in trades_df.columns and len(trades_df) > 0:
                    total_realized = trades_df['profit'].sum()
                    metrics['realized_pnl'] = f"${total_realized:+,.0f}"
        except:
            pass
        
        # Get unrealized P&L from current positions
        positions = get_positions()
        if positions and len(positions) > 0:
            df_positions = pd.DataFrame(positions)
            
            # Calculate unrealized P&L from current positions
            if all(col in df_positions.columns for col in ['current_price', 'entry_price', 'shares']):
                df_positions['unrealized_pnl'] = (
                    (df_positions['current_price'] - df_positions['entry_price']) * df_positions['shares']
                )
                total_unrealized = df_positions['unrealized_pnl'].sum()
                metrics['unrealized_pnl'] = f"${total_unrealized:+,.0f}"
            elif 'profit' in df_positions.columns:
                # Use profit column if available
                total_unrealized = df_positions['profit'].sum()
                metrics['unrealized_pnl'] = f"${total_unrealized:+,.0f}"
        
        # CORRECTED: Portfolio value = Starting Account + Total P&L (realized + unrealized)
        total_pnl = total_realized + total_unrealized
        total_portfolio_value = account_size + total_pnl
        metrics['total_value'] = f"${total_portfolio_value:,.0f}"
        
        # Calculate return percentage based on total P&L
        return_pct = (total_pnl / account_size * 100) if account_size > 0 else 0.0
        metrics['total_return_pct'] = f"{return_pct:+.2f}%"
        
        return metrics
        
    except Exception as e:
        st.error(f"Error getting enhanced portfolio metrics: {e}")
        return {
            'total_value': f"${account_size:,.0f}",
            'total_return_pct': "+0.0%",
            'daily_pnl': "$0.00",
            'unrealized_pnl': "$0.00",
            'realized_pnl': "$0.00",
            'win_rate': "0.0%"
        }

# Load enhanced metrics
metrics = get_enhanced_portfolio_metrics(account_size)

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

# Calculate L/S ratio
ls_ratio = calculate_ls_ratio()

# Portfolio Summary
st.subheader("Portfolio Summary")

# Custom CSS for metric fonts - IMPROVED SPACING v2.2
st.markdown("""
<style>
/* Improved spacing and font sizing v2.2 */
[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.1);
    padding: 2% 2% 2% 4% !important;
    border-radius: 5px;
    margin: 0 !important;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
    font-size: 0.65rem !important;
    font-weight: 600;
    white-space: nowrap !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 0.9rem !important;
    font-weight: 700;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.7rem !important;
    white-space: nowrap !important;
}

/* Additional spacing fixes */
.stMetric {
    margin: 0 !important;
    padding: 0 !important;
}

.stMetric > div {
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Now using 6 columns for portfolio summary (removed refresh button)
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
    st.metric(label="L/S Ratio", value=ls_ratio)

st.markdown("---")

# Current Positions - FIXED VERSION
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
        
        # FIXED: Ensure entry_date is properly formatted as string
        try:
            # Convert entry_date to datetime first, then to string
            df_positions['entry_date'] = pd.to_datetime(df_positions['entry_date']).dt.strftime('%Y-%m-%d')
        except:
            # If conversion fails, ensure it's at least string
            df_positions['entry_date'] = df_positions['entry_date'].astype(str)
        
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
        
        # Display positions count with L/S breakdown
        total_positions = len(display_df)
        long_count = len(df_positions[df_positions['side'] == 'long'])
        short_count = len(df_positions[df_positions['side'] == 'short'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Positions", total_positions)
        with col2:
            st.metric("Long Positions", long_count)
        with col3:
            st.metric("Short Positions", short_count)
        with col4:
            st.metric("L/S Ratio", ls_ratio)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Entry Date": st.column_config.TextColumn("Entry Date", width="small"),
                "Side": st.column_config.TextColumn("Side", width="small"),
                "Shares": st.column_config.NumberColumn("Shares", width="small"),
                "Entry Price": st.column_config.TextColumn("Entry Price", width="small"),
                "Current Price": st.column_config.TextColumn("Current Price", width="small"),
                "Current Value": st.column_config.TextColumn("Current Value", width="small"),
                "P&L": st.column_config.TextColumn("P&L", width="small"),
                "P&L %": st.column_config.TextColumn("P&L %", width="small")
            }
        )
        
        # FIXED: Check for suspicious future dates - now safe to use
        try:
            future_positions = df_positions[pd.to_datetime(df_positions['entry_date']) > datetime.now()]
            if not future_positions.empty:
                st.warning(f"⚠️ WARNING: {len(future_positions)} positions have FUTURE entry dates - these are likely from backtest data!")
                st.info("🔍 If you see future dates (like 04/09/25), these positions are synthetic/test data, not real trades.")
        except Exception as e:
            st.warning(f"Could not check for future dates: {e}")
        
        # FIXED: Check for very recent positions (today) - now safe to use string comparison
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            today_positions = df_positions[df_positions['entry_date'] == today]  # Use == instead of .str.contains()
            if not today_positions.empty:
                st.success(f"✅ {len(today_positions)} positions entered TODAY - these appear to be new live trades!")
        except Exception as e:
            st.warning(f"Could not check for today's positions: {e}")
    
    else:
        st.info("No current positions")
        # Show L/S ratio as N/A when no positions
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Positions", 0)
        with col2:
            st.metric("Long Positions", 0)
        with col3:
            st.metric("Short Positions", 0)
        with col4:
            st.metric("L/S Ratio", "N/A")

except Exception as e:
    st.error(f"Error loading positions: {e}")
    st.info("Check your data sources and ensure the system is properly configured.")
    # Add debug info
    import traceback
    st.code(traceback.format_exc())

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