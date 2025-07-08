# Add this function to your daily_update.py to replace the existing get_polygon_daily_data

def get_polygon_daily_data_with_indicators(symbol, days):
    """
    Download daily data and calculate indicators before saving.
    This ensures your rolling database always has complete data.
    """
    from nGS_Strategy import NGSStrategy
    from data_manager import load_price_data
    
    api_key = os.getenv('POLYGON_API_KEY') or "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"
    
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY environment variable is not set.")
    
    # Calculate dates with buffer for weekends
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 5)
    
    # Format dates for Polygon API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if "results" in data and data["results"]:
                # Convert API response to DataFrame
                new_df = pd.DataFrame(data["results"])
                new_df["Date"] = pd.to_datetime(new_df["t"], unit="ms")
                new_df.rename(columns={
                    "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
                }, inplace=True)
                new_df = new_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                new_df = new_df.sort_values("Date").tail(days)
                
                # Load existing data for indicator calculation context
                try:
                    existing_df = load_price_data(symbol)
                    if not existing_df.empty:
                        # Remove overlapping dates and combine
                        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                        new_dates = set(new_df['Date'].dt.date)
                        existing_df = existing_df[~existing_df['Date'].dt.date.isin(new_dates)]
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
                    else:
                        combined_df = new_df.copy()
                except:
                    combined_df = new_df.copy()
                
                # Calculate indicators
                strategy = NGSStrategy()
                df_with_indicators = strategy.calculate_indicators(combined_df)
                
                if df_with_indicators is not None and not df_with_indicators.empty:
                    logger.info(f"Downloaded {len(new_df)} rows + calculated indicators for {symbol}")
                    return df_with_indicators
                else:
                    logger.warning(f"Failed to calculate indicators for {symbol}, returning raw data")
                    return combined_df
            else:
                logger.warning(f"No results in API response for {symbol}")
                
        elif response.status_code == 429:
            logger.warning(f"Rate limit hit for {symbol}")
            return pd.DataFrame()
        else:
            logger.error(f"API error {response.status_code} for {symbol}: {response.text}")
            
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
    
    return pd.DataFrame()

# Replace the function call in your main daily update process
def download_data_parallel_with_indicators(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download data for multiple symbols in parallel with indicators."""
    data = {}
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # Use the new function that includes indicators
        future_to_symbol = {
            executor.submit(get_polygon_daily_data_with_indicators, symbol, CONFIG["history_days"]): symbol 
            for symbol in symbols
        }
        
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if not df.empty:
                    data[symbol] = df
                    # Save the data with indicators
                    save_price_data(symbol, df)
                    logger.info(f"Processed {symbol}: {len(df)} rows with indicators")
            except Exception as e:
                logger.error(f"Error processing {symbol} in parallel download: {e}")
    
    return data