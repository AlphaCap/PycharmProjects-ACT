import streamlit as st
from data_manager import get_trades_history_formatted, get_portfolio_metrics, get_positions
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Main Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit elements
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

# Sidebar
with st.sidebar:
    st.title("Trading Systems")
    st.caption(f"Last Updated: {datetime.now().strftime('%m/%d/%Y %H:%M')}")
    st.markdown("---")
    if st.button("Go to nGS Performance", use_container_width=True):
        try:
            st.switch_page("pages/1_nGS_System.py")
        except streamlit.errors.StreamlitAPIException:
            st.warning("Failed to switch to nGS page. Ensure 'pages/1_nGS_System.py' exists.")

# Main content
st.title("Main Dashboard")

# Initialize session state for account size
if 'initial_value' not in st.session_state:
    st.session_state.initial_value = 1000000
initial_value = st.number_input("Set initial portfolio/account size:", min_value=1000, value=st.session_state.initial_value, step=1000, format="%d", key="account_size_input_home")
st.session_state.initial_value = initial_value

# Fetch portfolio metrics with fallback
def get_portfolio_metrics_with_fallback(initial_value: int) -> dict:
    try:
        return get_portfolio_metrics(initial_portfolio_value=initial_value)
    except Exception as e:
        st.error(f"Error getting portfolio metrics: {e}")
        return {
            'total_value': f"${initial_value:,.0f}",
            'total_return_pct': "+0.0%",
            'daily_pnl': "$0.00",
            'me_ratio': "0.00",
            'mtd_return': "+0.0%",
            'ytd_return': "+0.0%"
        }

metrics = get_portfolio_metrics_with_fallback(initial_value)

# Ensure all required metrics exist with safe defaults
safe_metrics = {
    'total_value': f"${initial_value:,.0f}",
    'total_return_pct': "+0.0%",
    'daily_pnl': "$0.00",
    'me_ratio': "0.00",
    'mtd_return': "+0.0%",
    'ytd_return': "+0.0%"
}
for key, default_value in safe_metrics.items():
    if key not in metrics:
        metrics[key] = default_value

# Fetch positions for L/S Ratio
positions = get_positions()
long_positions = sum(pos['shares'] for pos in positions if pos.get('side', 'long') == 'long' and 'shares' in pos)
short_positions = sum(abs(pos['shares']) for pos in positions if pos.get('side', 'short') == 'short' and 'shares' in pos)
ls_ratio = (long_positions / short_positions) if short_positions > 0 else float('inf') if long_positions > 0 else 0.0
ls_ratio_display = f"{ls_ratio:.2f}" if ls_ratio != float('inf') else "∞" if long_positions > 0 else "0.00"

# Display Detailed Portfolio Metrics
st.subheader("📈 Detailed Portfolio Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_value_clean = str(metrics['total_value']).replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])
with col2:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])
with col3:
    st.metric(label="M/E Ratio", value=f"{metrics['me_ratio']}%")
with col4:
    st.metric(label="MTD Return", value=metrics['mtd_return'])  # Removed delta to avoid duplicate percentage
with col5:
    st.metric(label="L/S Ratio", value=ls_ratio_display)

# Second row
col6, col7 = st.columns(2)
with col6:
    st.metric(label="YTD Return", value=metrics['ytd_return'])
with col7:
    if st.button("🔄 Refresh Historical Data", use_container_width=True, key="refresh_button"):
        st.cache_data.clear()
        st.rerun()

# Display Trades
st.markdown("---")
st.subheader("📋 Trade History")
trades = get_trades_history_formatted()
st.dataframe(trades, use_container_width=True, hide_index=True)

# Display Positions (without redundant rows)
st.markdown("---")
st.subheader("📊 Current Positions")
positions_df = pd.DataFrame(positions)
if not positions_df.empty:
    st.dataframe(positions_df[['symbol', 'shares', 'entry_price', 'entry_date', 'current_price', 'profit', 'profit_pct', 'days_held', 'side']], use_container_width=True, hide_index=True)
else:
    st.info("No current positions available.")

# System Status
st.markdown("---")
st.subheader("⚙️ System Status")
try:
    st.success("✅ System Online")
    st.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
except Exception as e:
    st.error(f"Error getting system status: {e}")

st.markdown("<p style='text-align: center; color: #999; font-size: 0.8rem;'>* Data retention: 6 months (180 days)</p>", unsafe_allow_html=True)