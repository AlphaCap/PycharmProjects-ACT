import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Function to fetch all S&P 500 stock symbols and company names
def get_sp500_tickers():
    # Fetch S&P 500 data from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    
    # Extract the stock symbols and company names
    table = data[0]
    tickers = table['Symbol'].tolist()
    company_names = table['Security'].tolist()
    
    # Create a dictionary with ticker symbols as keys and company names as values
    ticker_company_dict = dict(zip(tickers, company_names))
    return ticker_company_dict

# Fetch historical prices of the S&P 500 index
def get_sp500_prices(start_date, end_date):
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    sp500_prices = sp500_data['Adj Close']
    return sp500_prices

# Sample function to optimize the portfolio using the Efficient Frontier
def optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount):
    n = len(selected_tickers)
    
    # Get adjusted close prices for the selected stocks
    my_portfolio = pd.DataFrame()
    for ticker in selected_tickers:
        my_portfolio[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    

    # Calculate daily returns for my_portfolio
    my_portfolio_returns = my_portfolio.pct_change().dropna()

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(my_portfolio)
    S = risk_models.sample_cov(my_portfolio)
    
    # Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=2)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Get the latest prices for discrete allocation
    latest_prices = get_latest_prices(my_portfolio)
    
    # Perform discrete allocation
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_amount)
    allocation, leftover = da.lp_portfolio()
    
    return my_portfolio_returns, cleaned_weights, latest_prices, allocation, leftover

def main():
    # Page configuration
    st.set_page_config(page_title="Stock Portfolio Optimizer")

    # Title and description
    st.title("Stock Portfolio Optimizer")
    st.write("Backtest your portfolio and get discrete allocation of stocks in the optimized portfolio.")
    
    st.markdown("---")
    st.subheader("Description")
    st.info(
            "This portfolio optimizer uses the Efficient Frontier method to optimize your portfolio based on historical stock prices from the S&P 500. "
            "You can select multiple stocks from the S&P 500, specify the time frame for historical data, and enter the amount you want to invest in the portfolio. "
            "The optimizer will then compute the optimal allocation of your investment across the selected stocks, aiming to maximize the portfolio's Sharpe ratio. "
            "Additionally, it will display a pie chart showing the allocation of each stock in the portfolio, along with a time series chart comparing the cumulative return of the optimized portfolio with the S&P 500's return."
        )
    st.markdown("---")

    # Sidebar for user inputs
    input_col = st.sidebar
    input_col.header("Input Timeframe")

    # User inputs
    start_date = input_col.date_input("Enter start date:", dt.datetime(2016, 1, 1))
    end_date = input_col.date_input("Enter end date:", dt.datetime.now())

    # Fetch all S&P 500 stock symbols and company names
    ticker_company_dict = get_sp500_tickers()

    input_col.header("Stock Portfolio")
    # Stock symbols dropdown with company names
    selected_tickers = input_col.multiselect("Select stock symbols:", list(ticker_company_dict.keys()), format_func=lambda ticker: f"{ticker}: {ticker_company_dict[ticker]}")

    # Portfolio amount
    portfolio_amount = input_col.number_input("Enter the investment amount:", min_value=1000.0, step=1000.0, value=100000.0, format="%.2f")

    # Optimization button
    if input_col.button("Optimize Portfolio"):
        if len(selected_tickers) < 2:
            st.warning("Please select multiple stock symbols.")
        else:
            my_portfolio_returns, cleaned_weights, latest_price, allocation, leftover = optimize_portfolio(selected_tickers, start_date, end_date, portfolio_amount)
            
            # Create a DataFrame to display the optimized portfolio allocation with cost and latest stock price
            df_allocation = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
            df_allocation['Stock Price'] = '$' + latest_price.round(2).astype(str)
            df_allocation['Cost'] = '$' + (df_allocation['Shares'] * latest_price).round(2).astype(str)
    
           # Create two columns for layout
            col1, col2 = st.columns([2, 2.5])

            # Display the optimized portfolio allocation table
            with col1:
                st.write("Discrete Allocation:")
                st.dataframe(df_allocation)
                st.write("Funds Remaining: ${:.2f}".format(leftover))

            # Create a pie chart
            with col2:
                st.write("Portfolio Composition:")
                # Set a custom color palette
                colors = sns.color_palette('Set3', len(df_allocation))

                # Explode the slice with the highest allocation
                explode = [0.05 if shares == max(df_allocation['Shares']) else 0 for shares in df_allocation['Shares']]

                # Plot the pie chart with custom styling
                plt.figure(figsize=(8,8))
                plt.pie(df_allocation['Shares'], labels=df_allocation.index, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
                plt.axis('equal')

                st.pyplot(plt)
            
            # Fetch historical prices of the S&P 500 index
            sp500_prices = get_sp500_prices(start_date, end_date)

            # Calculate daily returns of S&P 500
            sp500_returns = sp500_prices.pct_change().dropna()

            # Convert DataFrame to numpy array for dot product
            my_portfolio_returns_array = my_portfolio_returns.values
            cleaned_weights_array = np.array(list(cleaned_weights.values()))

            # Calculate portfolio returns using numpy dot product
            portfolio_returns = np.dot(my_portfolio_returns_array, cleaned_weights_array)

            # Calculate expected returns and volatility
            sp500_expected_returns = sp500_returns.mean() * 252  # Assuming 252 trading days in a year
            sp500_volatility = sp500_returns.std() * np.sqrt(252)
            portfolio_expected_returns = portfolio_returns.mean() * 252
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

            # Convert both to 1D numpy arrays
            sp500_returns_array = sp500_returns.values.flatten()
            portfolio_returns_array = portfolio_returns.flatten()
            
            # Combine the S&P 500 and portfolio returns into a DataFrame
            combined_returns = pd.DataFrame({'S&P 500': sp500_returns_array, 'Portfolio': portfolio_returns_array}, index=my_portfolio_returns.index)

            # Plot the time series chart of S&P 500 and the portfolio
            plt.figure(figsize=(12, 6))
            plt.plot(combined_returns.index, 100 * (combined_returns + 1).cumprod(), lw=2)
            plt.legend(combined_returns.columns)
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return (%)')
            plt.title('S&P 500 vs. Optimized Portfolio Performance')
            plt.grid(True)
            plt.tight_layout()

            # Display the chart
            st.pyplot(plt)

            df_info = pd.DataFrame({
                'S&P 500': ['{:.2f}%'.format(100 * sp500_expected_returns), '{:.2f}%'.format(100 * sp500_volatility)],
                'Portfolio': ['{:.2f}%'.format(100 * portfolio_expected_returns), '{:.2f}%'.format(100 * portfolio_volatility)]
            }, index=['Expected Return', 'Volatility'])

            st.dataframe(df_info)

            # Display a disclaimer at the bottom
            st.markdown("---")
            st.markdown("#### Disclaimer")
            st.info("The stock market involves inherent risks, and the performance of a portfolio is subject to market fluctuations. The information provided by this application is for educational and informational purposes only and does not constitute financial advice. Before making any investment decisions, it is advisable to conduct your own research or consult with a qualified financial advisor. The creator of this application shall not be liable for any losses or damages arising from the use of this application or the information provided herein.")
            st.markdown("---")

if __name__ == "__main__":
    main()
