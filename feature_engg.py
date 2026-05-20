# feature_engineering.py
import pandas as pd
import pandas_ta as ta
import config


def create_features(df_raw):
    """Generates technical and macro features (Stationary Version)."""
    df = df_raw.copy()

    print("Generating features...")

    # 1. Macro Changes (Stationarity Fix)
    # Instead of raw levels (e.g., 3.5% unemployment), we use the CHANGE (e.g., +0.1%)
    # We use diff() for rates and pct_change() for prices
    df['yield_spread_chg'] = (df['US_10YR_YIELD'] - df['IN_10YR_YIELD']).diff()
    df['inflation_spread_chg'] = (df['US_CPI'] - df['IN_CPI']).diff()

    # For Unemployment/Trade Balance, raw numbers are 'levels'. We need changes.
    df['us_unemp_chg'] = df['US_UNEMPLOYMENT'].diff()
    df['us_trade_chg'] = df['US_TRADE_BALANCE'].diff()

    # 2. Technicals
    df['returns'] = df[config.PRICE_COLUMN].pct_change()

    # Lags of returns (The most powerful feature usually)
    for lag in range(1, 4):
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # Relative Strength Index
    df['RSI'] = ta.rsi(df[config.PRICE_COLUMN], length=14)

    # Volatility
    df['volatility'] = df['returns'].rolling(30).std()

    # 3. Create Target (5-Day Horizon)
    df['future_price_5d'] = df[config.PRICE_COLUMN].shift(-5)
    df['target_direction'] = (df['future_price_5d'] > df[config.PRICE_COLUMN]).astype(int)

    # 4. Clean up
    # Drop the raw non-stationary columns to prevent confusion
    df = df.drop(columns=['US_10YR_YIELD', 'IN_10YR_YIELD', 'future_price_5d',
                          'US_UNEMPLOYMENT', 'US_TRADE_BALANCE', 'US_CPI', 'IN_CPI'], errors='ignore')

    df = df.dropna()

    return df