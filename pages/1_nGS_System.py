import streamlit as st
import streamlit.errors
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import (
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats,
    get_signals,
    get_system_status,
    get_trades_history
)

try:
    from portfolio_calculator import calculate_real_portfolio_metrics, get_enhanced_strategy_performance
    USE_REAL_METRICS = True
except ImportError:
    USE_REAL_METRICS = False

st.set_page_config(
    page_title="nGS Historical Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.title("Trading Systems")
    if st.button("← Back to Main Dashboard", use_container_width=True, key="main_dashboard_button"):
        try:
            st.switch_page("app.py")
        except streamlit.errors.StreamlitAPIException:
            st.warning("Failed to switch to main page. Ensure 'app.py' is the main script.")
            st.info("Current setup: Run 'streamlit run app.py' with this as a subpage.")

    st.markdown("---")
    st.caption(f"{datetime.now().strftime('%m/%d/%Y %H:%M')}")

st.markdown("### nGS Historical Performance")
st.caption("Detailed Performance Analytics & Trade History")

if 'initial_value' not in st.session_state:
    st.session_state.initial_value = 1000000
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=st.session_state.initial_value,
    step=1000,
    format="%d",
    key="account_size_input"
)
st.session_state.initial_value = initial_value

def get_portfolio_metrics_with_fallback(initial_value: int) -> dict:
    try:
        if USE_REAL_METRICS:
            return calculate_real_portfolio_metrics(initial_portfolio_value=initial_value)
        return get_portfolio_metrics(initial_portfolio_value=initial_value)
    except Exception as e:
        st.error(f"Error getting portfolio metrics: {e}")
        return {
            'total_value': f"${initial_value:,.0f}",
            'total_return_pct': "+0.0%",
            'daily_pnl': "$0.00",
            'me_ratio': "0.00",
            'mtd_return': "+0.0%",
            'mtd_delta': "+0.0%",
            'ytd_return': "+0.0%",
            'ytd_delta': "+0.0%"
        }

metrics = get_portfolio_metrics_with_fallback(initial_value)
safe_metrics = {
    'total_value': f"${initial_value:,.0f}",
    'total_return_pct': "+0.0%",
    'daily_pnl': "$0.00",
    'me_ratio': "0.00",
    'mtd_return': "+0.0%",
    'mtd_delta': "+0.0%",
    'ytd_return': "+0.0%",
    'ytd_delta': "+0.0%"
}
for key, default_value in safe_metrics.items():
    if key not in metrics:
        metrics[key] = default_value

st.subheader("📈 Detailed Portfolio Metrics")
if USE_REAL_METRICS and metrics.get('total_trades', 0) > 0:
    st.success(f"✅ Real portfolio metrics calculated from {metrics['total_trades']} trades")
    st.info(f"💰 Total profit: ${metrics.get('total_profit_raw', 0):,.2f} | Winners: {metrics.get('winning_trades', 0)} | Losers: {metrics.get('losing_trades', 0)}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_value_clean = str(metrics['total_value']).replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])
with col2:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])
with col3:
    historical_me = metrics.get('historical_me_ratio', '0.00')
    st.metric(label="Avg Historical M/E", value=f"{historical_me}%")
with col4:
    st.metric(label="MTD Return", value=metrics['mtd_return'], delta=metrics['mtd_delta'])

col5, col6 = st.columns([1, 1])
with col5:
    st.metric(label="YTD Return", value=metrics['ytd_return'], delta=metrics['ytd_delta'])
with col6:
    if st.button("🔄 Refresh Historical Data", use_container_width=True, key="refresh_button"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")
st.subheader("🎯 Strategy Performance")
def get_strategy_data(initial_value: int) -> pd.DataFrame:
    try:
        if USE_REAL_METRICS:
            return get_enhanced_strategy_performance(initial_portfolio_value=initial_value)
        return get_strategy_performance(initial_portfolio_value=initial_value)
    except Exception as e:
        st.error(f"Error loading strategy performance: {e}")
        return pd.DataFrame()
strategy_df = get_strategy_data(initial_value)
if not strategy_df.empty:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
else:
    st.info("No strategy performance data available.")

st.markdown("---")
st.subheader("⚠️ M/E Ratio Risk Management")
def plot_me_ratio_history(trades_df: pd.DataFrame, initial_value: int) -> None:
    try:
        from me_ratio_calculator import DailyMERatioCalculator
        if not trades_df.empty:
            calculator = DailyMERatioCalculator(initial_value)
            for _, row in trades_df.iterrows():
                calculator.update_position(row['symbol'], row['shares'], row['entry_price'], row['exit_price'], row['type'])
                # Convert string to Timestamp for comparison
                exit_date = pd.to_datetime(row['exit_date'])
                current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
                if exit_date <= current_date:
                    calculator.add_realized_pnl(row['profit'])
            me_history_df = calculator.get_me_history_df()
            if not me_history_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(me_history_df['Date'], me_history_df['ME_Ratio'], linewidth=3, color='#ff6b35', label='M/E Ratio', marker='o', markersize=4)
                ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.8, label='CRITICAL LIMIT (100%)')
                ax.fill_between(me_history_df['Date'], 0, 80, alpha=0.2, color='green', label='Safe Zone (<80%)')
                ax.fill_between(me_history_df['Date'], 80, 100, alpha=0.2, color='orange', label='Warning Zone (80-100%)')
                ax.set_title('M/E Ratio History - Critical for Portfolio Rebalancing', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('M/E Ratio (%)')
                ax.set_ylim(0, max(110, me_history_df['ME_Ratio'].max() * 1.1))
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')
                ax.tick_params(axis='x', rotation=45)
                avg_me = me_history_df['ME_Ratio'].mean()
                max_me = me_history_df['ME_Ratio'].max()
                min_me = me_history_df['ME_Ratio'].min()
                stats_text = f'Average: {avg_me:.1f}%\nMaximum: {max_me:.1f}%\nMinimum: {min_me:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                col1, col2, col3 = st.columns(3)
                with col1:
                    if max_me > 90:
                        st.error(f"🚨 HIGH RISK\nMax M/E: {max_me:.1f}%\n(>90% Critical)")
                    elif max_me > 80:
                        st.warning(f"⚠️ MODERATE RISK\nMax M/E: {max_me:.1f}%\n(>80% Warning)")
                    else:
                        st.success(f"✅ LOW RISK\nMax M/E: {max_me:.1f}%\n(<80% Safe)")
                with col2:
                    st.info(f"📊 **Average M/E Ratio**\n{avg_me:.1f}%\n(Historical Average)")
                with col3:
                    target_me = 75
                    if avg_me > target_me:
                        st.warning(f"🎯 **Rebalancing Signal**\nAvg: {avg_me:.1f}% > {target_me}%\nConsider reducing position sizes")
                    else:
                        st.success(f"🎯 **Portfolio Balanced**\nAvg: {avg_me:.1f}% ≤ {target_me}%\nWithin target range")
            else:
                st.info("No M/E ratio history available - need position data for analysis")
        else:
            st.info("No trade history for M/E ratio analysis")
    except Exception as e:
        st.error(f"Error creating M/E ratio analysis: {e}")

trades_df = get_trades_history()
plot_me_ratio_history(trades_df, initial_value)

st.markdown("---")
st.subheader("📊 Performance Statistics")
col1, col2 = st.columns([1, 1])
with col1:
    try:
        perf_stats_df = get_portfolio_performance_stats()
        if not perf_stats_df.empty:
            st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance statistics available.")
    except Exception as e:
        st.error(f"Error loading performance stats: {e}")
with col2:
    st.subheader("📈 Equity Curve")
    try:
        trades_df = get_trades_history()
        if not trades_df.empty:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_sorted = trades_df.sort_values('exit_date')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], linewidth=2, color='#1f77b4')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], where=(trades_sorted['cumulative_profit'] > 0), alpha=0.3, color='green')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], where=(trades_sorted['cumulative_profit'] <= 0), alpha=0.3, color='red')
            ax.set_title('Cumulative Profit Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Profit ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No trade history available for equity curve.")
    except Exception as e:
        st.error(f"Error creating equity curve: {e}")

st.markdown("---")
st.subheader("📋 Complete Trade History")
try:
    trades_df = get_trades_history()
    if not trades_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            st.metric("Winning Trades", winning_trades)
        with col3:
            win_rate = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            total_profit = trades_df['profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.2f}")
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History CSV",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history available.")
except Exception as e:
    st.error(f"Error loading trade history: {e}")

st.markdown("---")
st.subheader("🎯 Signal History")
try:
    signals_df = get_signals()
    if not signals_df.empty:
        st.dataframe(signals_df.head(50), use_container_width=True, hide_index=True)
        if len(signals_df) > 50:
            st.caption(f"Showing latest 50 signals out of {len(signals_df)} total")
    else:
        st.info("No signal history available.")
except Exception as e:
    st.error(f"Error loading signals: {e}")

st.markdown("---")
st.subheader("⚙️ System Status")
try:
    system_status = get_system_status()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ System Online")
        st.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        if USE_REAL_METRICS:
            st.success("✅ Real Metrics Active")
        else:
            st.warning("⚠️ Using Placeholder Metrics")
    with col3:
        st.info("📊 Data Sources Connected")
except Exception as e:
    st.error(f"Error getting system status: {e}")

st.markdown("<p style='text-align: center; color: #999; font-size: 0.8rem;'>* Data retention: 6 months (180 days)</p>", unsafe_allow_html=True)
st.markdown("---")
st.caption("nGulfStream Swing Trader - Historical Performance Analytics")
