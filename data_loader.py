# data_loader.py
import pandas as pd
import yfinance as yf
from fredapi import Fred
import config


def fetch_price_data(ticker, start, end, col_name):
    """Fetches daily closing price from yfinance (Force Clean)."""
    print(f"Fetching {ticker} price data from yfinance...")
    try:
        # Download data
        raw = yf.download(ticker, start=start, end=end, progress=False)

        # --- FIX: Handle yfinance MultiIndex Columns ---
        # yfinance often returns columns like ('Close', 'INR=X')
        # We extract just the 'Close' data, regardless of structure.

        if isinstance(raw.columns, pd.MultiIndex):
            # If multi-level, accessing 'Close' usually returns the inner column
            series = raw['Close']
        else:
            # If single-level, just get 'Close'
            series = raw['Close']

        # Ensure it's a Series (sometimes it returns a 1-col DataFrame)
        if isinstance(series, pd.DataFrame):
            # If it's still a DataFrame, take the first column
            series = series.iloc[:, 0]

        # --- Rebuild DataFrame from Scratch ---
        # This strips all 'levels' and metadata issues.
        price_df = pd.DataFrame(series.values, index=series.index, columns=[col_name])
        price_df.index.name = 'Date'

        # --- Timezone Clean ---
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)

        return price_df

    except Exception as e:
        print(f"Error fetching yfinance data: {e}")
        return None


def fetch_macro_data(api_key, tickers, start, end):
    """Fetches macro series from FRED."""
    print(f"Fetching {len(tickers)} macroeconomic series from FRED...")
    try:
        fred = Fred(api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}. Check FRED API key in config.py.")
        raise

    macro_data_list = []
    for code, name in tickers.items():
        try:
            series = fred.get_series(name, observation_start=start, observation_end=end)
            series.name = code
            macro_data_list.append(series)
        except Exception as e:
            # It's okay if some series fail (like discontinued ones)
            print(f"Warning: Could not fetch {name} ({code}).")

    # Combine all FRED series into a single DataFrame
    macro_df = pd.concat(macro_data_list, axis=1)
    return macro_df


def get_raw_data():
    """Main orchestrator to get and merge data."""
    # 1. Get Price
    price_df = fetch_price_data(config.PRICE_TICKER, config.START_DATE, config.END_DATE, config.PRICE_COLUMN)

    # 2. Get Macro
    macro_df = fetch_macro_data(config.FRED_API_KEY, config.FRED_TICKERS, config.START_DATE, config.END_DATE)

    # 3. Merge
    print("Merging data...")
    if price_df is not None and macro_df is not None:
        # Join on index (Date).
        # 'price_df' is now guaranteed to be a simple 1-level index.
        df_combined = price_df.join(macro_df, how='left')

        # Forward fill macro data (propagate last known value)
        df_final = df_combined.ffill().dropna()

        print(f"Data loading complete. Shape: {df_final.shape}")
        return df_final
    else:
        raise Exception("Data fetch failed.")