# config.py
# Global configuration and settings

# --- API KEYS ---
# !!! --- PASTE YOUR FRED API KEY BELOW --- !!!
FRED_API_KEY = '444988727ffbe3b9d834e4b3b486d657'

# --- DATA SETTINGS ---
START_DATE = '2015-01-01'
END_DATE = '2025-11-20'  # Use today's date or later

# --- TICKERS ---
PRICE_TICKER = 'INR=X'  # USD/INR
PRICE_COLUMN = 'USD_INR'

# Macroeconomic Data (FRED Codes)

# config.py (Update the FRED_TICKERS section)

FRED_TICKERS = {
    # Global Factors
    'OIL_PRICE': 'DCOILWTICO',
    'VIX': 'VIXCLS',

    # US Economy
    'US_10YR_YIELD': 'DGS10',
    'US_2YR_YIELD': 'DGS2',
    'US_CPI': 'CPIAUCSL',
    # 'US_GDP': 'GDP',  <-- COMMENT THIS OUT! It causes the "staircase effect"
    'US_UNEMPLOYMENT': 'UNRATE',
    'US_TRADE_BALANCE': 'BOPSTB',

    # India Economy
    'IN_10YR_YIELD': 'INDIRLTLT01STM',
    'IN_CPI': 'CPALTT01INM657N',
    'IN_TRADE_BALANCE': 'XTNTVA01INM664N'
}