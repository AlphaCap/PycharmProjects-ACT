import streamlit as st
import pandas as pd
import glob
import os
from datetime import datetime

def create_simple_dashboard():
    st.title("📊 nGS Trading Dashboard - Live Updates")
    
    # Scan for CSV files
    csv_files = glob.glob("*.csv")
    stock_files = [f for f in csv_files if len(f.replace('.csv', '')) <= 5]
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Updated", len(stock_files), "Real-time")
    
    with col2:
        if stock_files:
            latest_file = max(stock_files, key=os.path.getmtime)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
            st.metric("Last Update", mod_time.strftime("%H:%M:%S"), latest_file.replace('.csv', ''))
    
    with col3:
        st.metric("Total P&L", "$0.00", "Coming soon")
    
    with col4:
        st.metric("Active Trades", "0", "Detecting...")
    
    # Show recent data
    if stock_files:
        st.subheader("📊 Recent Market Data")
        
        for file in stock_files[:5]:
            try:
                df = pd.read_csv(file)
                symbol = file.replace('.csv', '').upper()
                
                if not df.empty and 'Close' in df.columns:
                    latest = df.iloc[-1]
                    previous = df.iloc[-2] if len(df) > 1 else latest
                    
                    price_change = latest['Close'] - previous['Close']
                    change_pct = (price_change / previous['Close']) * 100
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    
                    with col_a:
                        st.write(f"**{symbol}**")
                    
                    with col_b:
                        st.metric(
                            "Price", 
                            f"${latest['Close']:.2f}",
                            f"{change_pct:+.1f}%"
                        )
                    
                    with col_c:
                        if 'Volume' in latest:
                            st.write(f"Vol: {latest['Volume']:,.0f}")
                        
            except Exception as e:
                st.error(f"Error reading {file}: {e}")

if __name__ == "__main__":
    create_simple_dashboard()
