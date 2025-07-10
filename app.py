import streamlit as st
import pandas as pd
import datetime
import glob
import os
import numpy as np
from data_manager import (
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats,
    get_positions,
    get_long_positions_formatted,
    get_short_positions_formatted,
    get_signals,
    get_system_status,
    get_trades_history
)

# Import nGS System (Polygon data)
try:
    from nGS_Strategy import NGSStrategy
    from ngs_main_runner import NGSSystemRunner
    NGS_AVAILABLE = True
except ImportError:
    NGS_AVAILABLE = False

# Import the real portfolio calculator
try:
    from portfolio_calculator import calculate_real_portfolio_metrics, get_enhanced_strategy_performance, patch_portfolio_metrics
    # Patch the data_manager to use real calculations
    patch_portfolio_metrics()
    USE_REAL_METRICS = True
except ImportError:
    USE_REAL_METRICS = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Alpha Capture Technology AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIDE STREAMLIT ELEMENTS ---
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
    
    /* Hide only the auto-generated page navigation, not our custom sidebar */
    [data-testid="stSidebarNav"] {display: none;}
    
    .stAppViewContainer > .main .block-container {
        padding-top: 1rem;
    }
    
    /* Adjust metric font sizes for better fit - STRONGER OVERRIDE */
    div[data-testid="metric-container"] {
        font-size: 0.7rem !important;
    }
    
    div[data-testid="metric-container"] > div {
        font-size: 0.7rem !important;
    }
    
    div[data-testid="metric-container"] label {
        font-size: 0.65rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 100% !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        line-height: 1.1 !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-delta"] {
        font-size: 0.7rem !important;
    }
    
    /* Force smaller metrics specifically */
    .stMetric > div {
        font-size: 0.7rem !important;
    }
    
    .stMetric label {
        font-size: 0.65rem !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 0.9rem !important;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- LIVE UPDATE FUNCTIONS ---
@st.cache_data(ttl=30)  # Cache for 30 seconds
def scan_for_live_updates():
    """Scan for updated data and detect potential trades - nGS integration"""
    update_info = {
        'total_files': 0,
        'last_update': None,
        'latest_symbol': None,
        'market_moves': [],
        'potential_signals': []
    }
    
    # Try to get nGS data first
    if NGS_AVAILABLE:
        try:
            runner = NGSSystemRunner()
            results = runner.run()
            
            if results and 'current_signals' in results:
                update_info['potential_signals'] = results['current_signals']
                update_info['total_files'] = results.get('symbols_scanned', 0)
                update_info['last_update'] = datetime.datetime.now()
                update_info['latest_symbol'] = 'nGS_SCAN'
                
        except Exception as e:
            # Fallback to CSV scanning if nGS fails
            pass
    
    # Fallback to CSV scanning if nGS not available or failed
    if not update_info['potential_signals']:
        csv_files = glob.glob("*.csv")
        stock_files = [f for f in csv_files if len(f.replace('.csv', '')) <= 5 and f.replace('.csv', '').isalpha()]
        
        update_info['total_files'] = len(stock_files)
        
        if stock_files:
            # Find most recently updated file
            latest_file = max(stock_files, key=os.path.getmtime)
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(latest_file))
            update_info['last_update'] = mod_time
            update_info['latest_symbol'] = latest_file.replace('.csv', '').upper()
            
            # Scan for significant market moves
            for file in stock_files[:10]:  # Check first 10 files
                try:
                    df = pd.read_csv(file)
                    symbol = file.replace('.csv', '').upper()
                    
                    if not df.empty and len(df) >= 2 and 'Close' in df.columns:
                        latest = df.iloc[-1]
                        previous = df.iloc[-2]
                        
                        price_change = latest['Close'] - previous['Close']
                        change_pct = (price_change / previous['Close']) * 100
                        
                        # Track significant moves (>1%)
                        if abs(change_pct) > 1.0:
                            update_info['market_moves'].append({
                                'symbol': symbol,
                                'price': latest['Close'],
                                'change_pct': change_pct,
                                'volume': latest.get('Volume', 0)
                            })
                        
                        # Detect potential trading signals
                        signal = detect_trading_signal(df, symbol)
                        if signal:
                            update_info['potential_signals'].append(signal)
                            
                except Exception:
                    continue
    
    return update_info

def detect_trading_signal(df, symbol):
    """Simple signal detection based on technical indicators"""
    if len(df) < 2:
        return None
        
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    signals = []
    
    # Bollinger Band signals
    if all(col in latest.index for col in ['Close', 'UpperBB', 'LowerBB', 'BBAvg']):
        if latest['Close'] > latest['UpperBB'] and previous['Close'] <= previous['UpperBB']:
            signals.append({
                'symbol': symbol,
                'type': 'BB_Breakout_Up',
                'price': latest['Close'],
                'strength': 'Strong',
                'direction': 'LONG'
            })
        elif latest['Close'] < latest['LowerBB'] and previous['Close'] >= previous['LowerBB']:
            signals.append({
                'symbol': symbol,
                'type': 'BB_Breakout_Down', 
                'price': latest['Close'],
                'strength': 'Strong',
                'direction': 'SHORT'
            })
    
    # PSAR signals
    if 'PSAR_IsLong' in latest.index:
        if latest['PSAR_IsLong'] == 1 and previous.get('PSAR_IsLong', 0) == 0:
            signals.append({
                'symbol': symbol,
                'type': 'PSAR_Long',
                'price': latest['Close'],
                'strength': 'Medium',
                'direction': 'LONG'
            })
        elif latest['PSAR_IsLong'] == 0 and previous.get('PSAR_IsLong', 1) == 1:
            signals.append({
                'symbol': symbol,
                'type': 'PSAR_Short',
                'price': latest['Close'],
                'strength': 'Medium', 
                'direction': 'SHORT'
            })
    
    # Linear Regression trend
    if 'oLRSlope' in latest.index and 'LinReg' in latest.index:
        slope = latest['oLRSlope']
        if abs(slope) > 0.5:  # Strong trend
            if slope > 0 and latest['Close'] > latest['LinReg']:
                signals.append({
                    'symbol': symbol,
                    'type': 'Strong_Uptrend',
                    'price': latest['Close'],
                    'strength': 'Strong' if abs(slope) > 1.0 else 'Medium',
                    'direction': 'LONG'
                })
            elif slope < 0 and latest['Close'] < latest['LinReg']:
                signals.append({
                    'symbol': symbol,
                    'type': 'Strong_Downtrend',
                    'price': latest['Close'],
                    'strength': 'Strong' if abs(slope) > 1.0 else 'Medium',
                    'direction': 'SHORT'
                })
    
    # Return strongest signal
    if signals:
        return max(signals, key=lambda x: 1 if x['strength'] == 'Strong' else 0)
    
    return None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Trading Systems")
    if st.button("nGulfStream Swing Trader", use_container_width=True):
        st.switch_page("pages/1_nGS_System.py")
    
    # nGS System status indicator
    ngs_status = "🟢 Active" if NGS_AVAILABLE else "🔴 Offline"
    st.caption(f"nGS System: {ngs_status}")
    
    # Disabled placeholder buttons for future systems
    st.button("Alpha Capture AI", use_container_width=True, disabled=True, help="Coming Soon")
    st.button("gST DayTrader", use_container_width=True, disabled=True, help="Coming Soon")
    
    st.markdown("---")
    st.caption("Data last updated:")
    st.caption(f"{datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}")
    
    # Add live update status in sidebar
    st.markdown("---")
    st.subheader("🔄 Live Updates")
    
    live_data = scan_for_live_updates()
    
    if live_data['last_update']:
        st.success(f"✅ Data current")
        st.caption(f"Last: {live_data['last_update'].strftime('%H:%M:%S')}")
        st.caption(f"Symbol: {live_data['latest_symbol']}")
    else:
        st.warning("⚠️ No recent updates")
    
    st.metric("Files Tracked", live_data['total_files'])
    st.metric("Market Moves", len(live_data['market_moves']))

# --- PAGE HEADER ---
st.title("Alpha Capture Technology AI")
st.caption("S&P 500 Long/Short Position Trader")

# Data source indicator
if NGS_AVAILABLE:
    st.success("📊 Data Source: nGS System (Polygon API)")
else:
    st.warning("📊 Data Source: CSV Files (nGS System offline)")

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Current Portfolio Status")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- PORTFOLIO METRICS - SINGLE LINE ONLY ---
if USE_REAL_METRICS:
    metrics = calculate_real_portfolio_metrics(initial_portfolio_value=initial_value)
else:
    metrics = get_portfolio_metrics(initial_portfolio_value=initial_value)

# Ensure all required metrics exist with defaults
required_metrics = {
    'total_value': f"${initial_value:,.0f}",
    'total_return_pct': "+0.0%",
    'daily_pnl': "$0.00",
    'me_ratio': "0.00",
    'mtd_return': "+0.0%",
    'ytd_return': "+0.0%"
}

# Fill in missing metrics with defaults
for key, default_value in required_metrics.items():
    if key not in metrics:
        metrics[key] = default_value

# Portfolio Overview Metrics - Single Row, 7 Metrics + Refresh Button
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    total_value_clean = str(metrics['total_value']).replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean)
with col2:
    st.metric(label="Total Return", value=metrics['total_return_pct'])
with col3:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])
with col4:
    st.metric(label="M/E Ratio", value=metrics['me_ratio'])
with col5:
    st.metric(label="MTD Return", value=metrics['mtd_return'])
with col6:
    st.metric(label="YTD Return", value=metrics['ytd_return'])
with col7:
    # Auto-refresh toggle
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --- CURRENT POSITIONS SECTION ---
st.markdown("## Current Positions")

# Long Positions
st.subheader("📈 Long Positions")
long_positions_df = get_long_positions_formatted()
if not long_positions_df.empty:
    st.dataframe(long_positions_df, use_container_width=True, hide_index=True)
    
    # Long positions summary
    long_count = len(long_positions_df)
    long_total_value = long_positions_df['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
    st.caption(f"**Long Summary:** {long_count} positions, Total P&L: ${long_total_value:.2f}")
else:
    st.info("No active long positions.")

# Short Positions  
st.subheader("📉 Short Positions")
short_positions_df = get_short_positions_formatted()
if not short_positions_df.empty:
    st.dataframe(short_positions_df, use_container_width=True, hide_index=True)
    
    # Short positions summary
    short_count = len(short_positions_df)
    short_total_value = short_positions_df['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
    st.caption(f"**Short Summary:** {short_count} positions, Total P&L: ${short_total_value:.2f}")
else:
    st.info("No active short positions.")

# --- TODAY'S SIGNALS SECTION ---
st.markdown("## Today's Signals")

# Get signals from data manager
signals_df = get_signals()
if not signals_df.empty:
    # Filter for today's signals if date column exists
    today = datetime.datetime.now().date()
    if 'date' in signals_df.columns:
        signals_df['date'] = pd.to_datetime(signals_df['date']).dt.date
        todays_signals = signals_df[signals_df['date'] == today]
    else:
        todays_signals = signals_df.head(10)  # Show recent signals
    
    if not todays_signals.empty:
        # Display today's signals in a clean format
        for _, signal in todays_signals.iterrows():
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
            
            with col_a:
                st.write(f"**{signal['symbol']}**")
            with col_b:
                direction_emoji = "🟢" if "long" in str(signal['signal_type']).lower() else "🔴"
                st.write(f"{direction_emoji} {signal['signal_type']}")
            with col_c:
                st.write(f"${signal['price']}")
            with col_d:
                st.write(f"{signal['strategy']}")
    else:
        st.info("No signals generated today.")
else:
    st.info("No recent signals available.")

# Add live signals if available
live_data = scan_for_live_updates()
if live_data['potential_signals']:
    st.subheader("⚡ Live Signals")
    
    for signal in live_data['potential_signals'][:5]:  # Show top 5 live signals
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
        
        with col_a:
            st.write(f"**{signal['symbol']}**")
        with col_b:
            direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
            strength_emoji = "🔥" if signal['strength'] == 'Strong' else "⚡"
            st.write(f"{direction_emoji} {signal['direction']}")
        with col_c:
            st.write(f"${signal['price']:.2f}")
        with col_d:
            st.write(f"{strength_emoji} {signal['type'].replace('_', ' ')}")

st.markdown("---")
st.caption("Alpha Trading Systems Dashboard - For additional support, please contact the trading desk.")