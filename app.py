import streamlit as st
from data_manager import get_trades_history_formatted, get_portfolio_metrics, get_positions

st.title("Main Dashboard")
if 'initial_value' not in st.session_state:
    st.session_state.initial_value = 1000000
initial_value = st.number_input("Set initial portfolio/account size:", min_value=1000, value=st.session_state.initial_value, step=1000, format="%d", key="account_size_input_home")
st.session_state.initial_value = initial_value

metrics = get_portfolio_metrics(initial_value)
st.write("Portfolio Metrics:", metrics)
trades = get_trades_history_formatted()
st.dataframe(trades)
positions = get_positions()
st.write("Current Positions:", positions)
