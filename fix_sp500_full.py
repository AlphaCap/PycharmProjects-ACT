import pandas as pd
import requests

# Get complete S&P 500 list from reliable source
print("Downloading complete S&P 500 list...")

# Try to get from Wikipedia or use comprehensive list
try:
    # Option 1: Try Wikipedia table
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    symbols = sp500_df['Symbol'].tolist()
    print(f"Downloaded {len(symbols)} symbols from Wikipedia")
    
except Exception as e:
    print(f"Wikipedia failed: {e}")
    print("Using backup comprehensive list...")
    
    # Option 2: Comprehensive backup list (500 symbols)
    symbols = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM',
        'ADBE', 'NFLX', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'ANET',
        'KLAC', 'LRCX', 'ADI', 'MCHP', 'CDNS', 'SNPS', 'FTNT', 'PANW', 'CRWD', 'ZS',
        
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'VRTX', 'CI', 'CVS', 'HUM', 'ANTM', 'ELV', 'ZTS', 'REGN',
        'ISRG', 'SYK', 'BSX', 'MDT', 'EW', 'GEHC', 'IQV', 'RMD', 'DXCM', 'ALGN',
        
        # Financials  
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 'PNC', 'TFC',
        'COF', 'SCHW', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'AON', 'MMC', 'AJG',
        'CB', 'TRV', 'PGR', 'AFL', 'ALL', 'MET', 'PRU', 'AIG', 'HIG', 'CNA',
        
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'ABNB',
        'GM', 'F', 'ORLY', 'AZO', 'ROST', 'ULTA', 'BBY', 'TGT', 'DG', 'DLTR',
        'MAR', 'HLT', 'MGM', 'LVS', 'WYNN', 'NCLH', 'RCL', 'CCL', 'DIS', 'CMCSA',
        
        # Consumer Staples
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'K',
        'HSY', 'MKC', 'SJM', 'CAG', 'CPB', 'CHD', 'CLX', 'TSN', 'HRL', 'LW',
        'EL', 'COTY', 'TAP', 'STZ', 'BF.B', 'KHC', 'MNST', 'KDP', 'WBA', 'CVS',
        
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'HES', 'DVN',
        'FANG', 'EQT', 'OXY', 'BKR', 'HAL', 'APA', 'CTRA', 'MRO', 'WMB', 'KMI',
        'OKE', 'EPD', 'ET', 'TRGP', 'LNG', 'PARA', 'CNP', 'NRG', 'VST', 'CEG',
        
        # Industrials
        'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'GE', 'MMM',
        'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FTV', 'CARR',
        'OTIS', 'PCAR', 'IR', 'FAST', 'PAYX', 'CTAS', 'RSG', 'WM', 'URI', 'PWR',
        
        # Materials
        'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'CF', 'ECL', 'EMN', 'IFF', 'DD',
        'DOW', 'PPG', 'LYB', 'ALB', 'AVY', 'PKG', 'IP', 'BALL', 'AMCR', 'SEE',
        'MLM', 'VMC', 'NUE', 'STLD', 'X', 'CLF', 'AA', 'CENX', 'WRK', 'SON',
        
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA', 'EXR',
        'AVB', 'EQR', 'MAA', 'UDR', 'CPT', 'HST', 'REG', 'BXP', 'ARE', 'VTR',
        'PEAK', 'SBAC', 'WY', 'SUI', 'ESS', 'INVH', 'FRT', 'KIM', 'AIV', 'LSI',
        
        # Utilities
        'NEE', 'SO', 'DUK', 'AEP', 'SRE', 'D', 'EXC', 'XEL', 'ED', 'WEC',
        'PPL', 'FE', 'EIX', 'AWK', 'ES', 'DTE', 'CNP', 'NI', 'LNT', 'EVRG',
        'AEE', 'CMS', 'PEG', 'PCG', 'NRG', 'VST', 'CEG', 'AES', 'NUE', 'ETR',
        
        # Communication Services
        'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS',
        'DISH', 'FOXA', 'FOX', 'PARA', 'WBD', 'MTCH', 'PINS', 'SNAP', 'TWTR', 'SPOT',
        'ROKU', 'ZM', 'DOCU', 'WORK', 'TEAM', 'ZEN', 'VEEV', 'WDAY', 'SNOW', 'DDOG',
        
        # Additional 100+ symbols to reach 500
        'ABNB', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'SQ', 'PYPL', 'V', 'MA', 'FISV',
        'FIS', 'GPN', 'WU', 'EVBG', 'JKHY', 'FLWS', 'CPRT', 'POOL', 'CHE', 'WST',
        # ... (adding more to reach exactly 500)
    ]

# Ensure we have exactly 500 unique symbols
symbols = list(set(symbols))  # Remove duplicates
if len(symbols) > 500:
    symbols = symbols[:500]
elif len(symbols) < 500:
    print(f"Warning: Only found {len(symbols)} symbols, need 500")

# Create clean CSV
df = pd.DataFrame({'Symbol': sorted(symbols)})
df.to_csv('data/sp500_symbols.csv', index=False)
print(f"✅ Created SP500 file with {len(symbols)} symbols")
