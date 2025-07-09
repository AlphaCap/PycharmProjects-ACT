import streamlit as st
from simple_ngs_dashboard import create_simple_dashboard

st.set_page_config(page_title="nGS Dashboard", layout="wide")
create_simple_dashboard()