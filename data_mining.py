import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_greeks(S, K, T, r, iv_sigma, q=0, option_type='call'):
    """
    Calculate Black-Scholes Greeks for a single option.
    
    Args:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate (decimal)
        sigma: Implied volatility (decimal)
        q: Dividend yield (decimal, default 0)
        option_type: 'call' or 'put'
    
    Returns:
        dict: Contains delta, gamma, theta, vega, rho
    """
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + iv_sigma**2 / 2) * T) / (iv_sigma * np.sqrt(T))
    d2 = d1 - iv_sigma * np.sqrt(T)
    
    # Common calculations
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * iv_sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in vol
    
    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * iv_sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2) 
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in rates
        
    elif option_type == 'put':
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * iv_sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change in rates
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Theta is typically expressed as the daily decay (divide by 365)
    theta_daily = theta / 365
    
    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta_daily': round(theta_daily, 4),
        'vega_1pct': round(vega, 4),
        'rho_1pct': round(rho, 4)
    }

def yf_option_chain(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    expirations = tk.options
    data = pd.DataFrame()

    for exp_td_str in expirations[:1]:
        options = tk.option_chain(exp_td_str)
        calls = options.calls
        puts = options.puts

        # Add optionType column
        calls['optionType'] = 'C'
        puts['optionType'] = 'P'

        # Merge calls and puts into a single dataframe
        exp_data = pd.concat(objs=[calls, puts], ignore_index=True)
        data = pd.concat(objs=[data, exp_data], ignore_index=True)

    # Add underlyingSymbol column
    data['underlyingSymbol'] = ticker

    return data


def yf_underlying_data(ticker: str, years: float | None) -> pd.DataFrame:
    # Calculate date range in years
    end_date = datetime.now()
    start_date = end_date - timedelta(days= years *365)  # Approx 30 years
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    # Download data for all tickers
    print(f"Downloading ({ticker})...")
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date) 
        # Keep only relevant columns and rename them
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        print(data.columns)
        print(f"  Successfully downloaded {len(data)} rows")

        if data is not None:
            df = data
            
    except Exception as e:
            print(f"  Error downloading {ticker}: {str(e)}")
    
    # Initial data info
    print(type(df))
    print(f"\nInitial data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 1: Remove any rows where all data is NaN (shouldn't happen with concat, but just in case)
    df = df.dropna(how='all')
    
    # Forward fill for up to 5 days (for holidays, etc.)
    df = df.fillna(method='ffill', limit=5)
    
    # Backward fill for the beginning if first few days are NaN
    df = df.fillna(method='bfill', limit=5)
    
    # Check for and handle any remaining NaNs
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"\nWarning: {remaining_nulls} NaN values remain after cleaning. Dropping these rows.")
        df = df.dropna()
    
    return df

def _private_save_data(df, filename='yahoo_finance_data.csv'):
    """
    Save the cleaned data to CSV file
    """
    if df is not None:
        df.to_csv(filename)
        print(f"\nData saved to {filename}")

def _private_run():
    folder = "data"

    tickers = ['SPY', '^VIX']
    
    for ticker in tickers:
        df = yf_underlying_data(ticker, years=25)
        if df is not None:
            _private_save_data(df, f"{folder}/{ticker}_data.csv")
    
        
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    _private_run()