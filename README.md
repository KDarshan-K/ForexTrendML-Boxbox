# 📈 ForexTrendML-Boxbox: USD/INR Macro-Quant System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered machine learning pipeline and interactive web application designed to forecast the **5-day directional movement** of the USD/INR exchange rate. It leverages historical market data, custom feature engineering, and macroeconomic indicators (from the Federal Reserve Economic Data - FRED) to generate confidence-based trading signals.

---

## ✨ Features

- **Interactive Streamlit Dashboard (`app.py`)**: A real-time web interface displaying live predictions, feature importance, and an interactive price projection chart with confidence intervals.
- **Financial Backtesting (`main.py`)**: Simulates a trading strategy against historical data using customizable confidence thresholds, calculating key metrics such as Total Strategy Return, Buy & Hold Return, and the Sharpe Ratio.
- **Robust Feature Engineering (`feature_engg.py`)**: Computes technical indicators (Moving Averages, Volatility, Momentum) alongside macroeconomic data to capture market trends.
- **Automated Data Pipeline (`data_loader.py`)**: Seamlessly fetches and merges financial data from Yahoo Finance and economic data from the FRED API.
- **Model Evaluation (`plots.py` & `model_train.py`)**: Trains an XGBoost classifier with built-in mechanisms to visualize model performance, generating confusion matrices and historical prediction graphs.

---

## 🛠️ Project Structure

```text
ForexTrendML-Boxbox/
│
├── app.py                   # Streamlit web dashboard
├── config.py                # Global configuration & API keys
├── data_loader.py           # Data extraction from yfinance & FRED
├── feature_engg.py          # Technical & macro feature engineering
├── main.py                  # Model backtesting & execution script
├── model_train.py           # ML training module (XGBoost)
├── plots.py                 # Utilities for evaluation graphs
├── confusion_matrix.png     # Example model performance plot
└── prediction_graph.png     # Example backtest visualization
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ForexTrendML-Boxbox.git
cd ForexTrendML-Boxbox
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages:
```bash
pip install pandas numpy xgboost scikit-learn streamlit plotly yfinance fredapi
```

### 3. API Key Configuration
This project relies on the **FRED API** for macroeconomic data.
1. Get a free API key from [FRED Services](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Open `config.py` and update the API key:
   ```python
   # config.py
   FRED_API_KEY = 'YOUR_FRED_API_KEY_HERE'
   ```

---

## 💻 Usage

### Running the Interactive Dashboard
To launch the real-time AI forecaster dashboard:
```bash
streamlit run app.py
```
This will open a local web server (typically at `http://localhost:8501`) where you can adjust confidence thresholds, view live signals, and analyze feature importance.

### Running the Financial Backtest
To execute the pipeline, train the model on historical data, and run the financial backtest simulation:
```bash
python main.py
```
*Output will include the threshold used, strategy total return, buy & hold return, and Sharpe ratio.*

---

## 📊 Model & Methodology
- **Target Variable**: The direction of the USD/INR closing price 5 days into the future (Up = 1, Down = 0).
- **Algorithm**: XGBoost Classifier (`XGBClassifier`)
- **Threshold Strategy**: To minimize false positives, the strategy only triggers a "Buy" or "Sell" signal if the model's confidence exceeds a specified threshold (e.g., >60%). Otherwise, it defaults to a "Hold".

---

