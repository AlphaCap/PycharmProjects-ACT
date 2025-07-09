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
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- LIVE UPDATE FUNCTIONS ---
@st.cache_data(ttl=30)  # Cache for 30 seconds
def scan_for_live_updates():
    """Scan for updated CSV files and detect potential trades"""
    csv_files = glob.glob("*.csv")
    stock_files = [f for f in csv_files if len(f.replace('.csv', '')) <= 5 and f.replace('.csv', '').isalpha()]
    
    update_info = {
        'total_files': len(stock_files),
        'last_update': None,
        'latest_symbol': None,
        'market_moves': [],
        'potential_signals': []
    }
    
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

# --- LIVE MARKET ACTIVITY BANNER ---
live_data = scan_for_live_updates()

if live_data['market_moves'] or live_data['potential_signals']:
    st.markdown("## 🔥 Live Market Activity")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if live_data['market_moves']:
            st.subheader("📊 Significant Moves (>1%)")
            
            moves_col1, moves_col2, moves_col3 = st.columns(3)
            
            for i, move in enumerate(live_data['market_moves'][:6]):  # Show top 6 moves
                col = [moves_col1, moves_col2, moves_col3][i % 3]
                
                with col:
                    change_color = "🟢" if move['change_pct'] > 0 else "🔴"
                    st.metric(
                        f"{change_color} {move['symbol']}", 
                        f"${move['price']:.2f}",
                        f"{move['change_pct']:+.1f}%"
                    )
    
    with col2:
        if live_data['potential_signals']:
            st.subheader("⚡ Live Signals")
            
            for signal in live_data['potential_signals'][:5]:  # Show top 5 signals
                direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
                strength_emoji = "🔥" if signal['strength'] == 'Strong' else "⚡"
                
                st.markdown(
                    f"{direction_emoji} **{signal['symbol']}** | "
                    f"{strength_emoji} {signal['strength']} | "
                    f"${signal['price']:.2f}"
                )
                st.caption(f"Signal: {signal['type'].replace('_', ' ')}")

    st.markdown("---")

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Current Portfolio Status")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- PORTFOLIO METRICS ---
metrics = get_portfolio_metrics(initial_portfolio_value=initial_value)

# Portfolio Overview Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Remove cents from total value display
    total_value_clean = metrics['total_value'].replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])
with col2:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])
with col3:
    st.metric(label="M/E Ratio", value=metrics['me_ratio'])
with col4:
    st.metric(label="Long Exposure", value=metrics['long_exposure'])
with col5:
    st.metric(label="Short Exposure", value=metrics['short_exposure'])

# Net Exposure
col6, col7, col8 = st.columns(3)
with col6:
    st.metric(label="Net Exposure", value=metrics['net_exposure'])
with col7:
    st.metric(label="MTD Return", value=metrics['mtd_return'], delta=metrics['mtd_delta'])
with col8:
    st.metric(label="YTD Return", value=metrics['ytd_return'], delta=metrics['ytd_delta'])

# --- STRATEGY PERFORMANCE TABLE ---
st.subheader("Strategy Performance")
strategy_df = get_strategy_performance(initial_portfolio_value=initial_value)
if not strategy_df.empty:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
else:
    st.info("No strategy performance data available.")

# --- POSITIONS SECTION ---
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

# --- SIGNALS AND PERFORMANCE SECTION ---
col1, col2 = st.columns([1, 2])  # Make signals column narrower

with col1:
    st.subheader("Today's Signals")
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
            for _, signal in todays_signals.iterrows():
                st.markdown(
                    f"- **{signal['date']}** | **{signal['symbol']}** | **{signal['signal_type']}** | {signal['strategy']} at `{signal['price']}`"
                )
        else:
            st.info("No signals generated today.")
    else:
        st.info("No recent signals.")

with col2:
    # Split this column for performance stats and equity curve
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        st.subheader("Portfolio Performance Stats")
        perf_stats_df = get_portfolio_performance_stats()
        if not perf_stats_df.empty:
            st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance statistics available.")
    
    with subcol2:
        st.subheader("Equity Curve")
        # Create equity curve from trade history
        trades_df = get_trades_history()
        if not trades_df.empty:
            try:
                # Sort trades by exit date and calculate cumulative profit
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                trades_sorted = trades_df.sort_values('exit_date')
                trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
                
                # Create the equity curve chart
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], linewidth=2, color='#1f77b4')
                ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                               where=(trades_sorted['cumulative_profit'] > 0), alpha=0.3, color='green')
                ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                               where=(trades_sorted['cumulative_profit'] <= 0), alpha=0.3, color='red')
                ax.set_title('Cumulative Profit Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Profit ($)')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating equity curve: {e}")
        else:
            st.info("No trade history for equity curve.")

# --- LIVE DATA MONITORING SECTION ---
st.markdown("---")
st.markdown("## 📊 Live Data Monitoring")

# Auto-refresh toggle
auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📁 Data Status")
    st.metric("CSV Files", live_data['total_files'])
    
    if live_data['last_update']:
        st.success(f"✅ Last update: {live_data['last_update'].strftime('%H:%M:%S')}")
        st.info(f"📊 Latest: {live_data['latest_symbol']}")
    else:
        st.warning("⚠️ No recent data updates")

with col2:
    st.subheader("🎯 Signal Summary")
    st.metric("Active Signals", len(live_data['potential_signals']))
    st.metric("Market Moves >1%", len(live_data['market_moves']))
    
    if live_data['potential_signals']:
        strong_signals = len([s for s in live_data['potential_signals'] if s['strength'] == 'Strong'])
        st.metric("Strong Signals", strong_signals)

with col3:
    st.subheader("⚡ System Status")
    st.success("✅ nGS System Online")
    st.info("🔄 Data Updating")
    
    if st.button("🔍 Refresh Data Now"):
        st.cache_data.clear()
        st.rerun()

# Auto-refresh functionality
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

st.markdown("---")
st.caption("Alpha Trading Systems Dashboard - For additional support, please contact the trading desk.")
