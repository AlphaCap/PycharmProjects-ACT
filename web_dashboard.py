import streamlit as st
from ngs_dashboard import create_ngs_dashboard

st.set_page_config(page_title="nGS Live Dashboard", layout="wide")

# Run the complete dashboard system
create_ngs_dashboard()
