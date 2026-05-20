# main.py
import data_loader
import feature_engg
import model_train
import pandas as pd
import numpy as np


# main.py (Only the backtest_strategy function needs changing)

def backtest_strategy(model, X_test, y_test, original_df):
    """
    Simulates a trading strategy with a Probability Threshold.
    """
    print("\n[STEP 4] Financial Backtesting (Weekly Horizon)...")

    # 1. Get Probabilities instead of just 0/1 predictions
    # proba[:, 1] is the probability of "Up"
    probs = model.predict_proba(X_test)[:, 1]

    # 2. Define Threshold (e.g., only trade if > 55% confident)
    CONFIDENCE_THRESHOLD = 0.6

    # Generate signals: 1 (Buy) if prob > 0.55, -1 (Sell) if prob < 0.45, else 0 (Hold)
    signals = np.where(probs > CONFIDENCE_THRESHOLD, 1,
                       np.where(probs < (1 - CONFIDENCE_THRESHOLD), -1, 0))

    # 3. Calculate Returns
    test_dates = X_test.index
    actual_returns = original_df.loc[test_dates, 'returns']

    # Strategy returns
    strategy_returns = signals * actual_returns

    # Cumulative Returns
    cum_market = (1 + actual_returns).cumprod()
    cum_strategy = (1 + strategy_returns).cumprod()

    # Metrics
    total_return = cum_strategy.iloc[-1] - 1
    market_return = cum_market.iloc[-1] - 1

    # Annualized Sharpe (Assuming daily data points)
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)

    print(f"Threshold Used:        {CONFIDENCE_THRESHOLD * 100}%")
    print(f"Strategy Total Return: {total_return * 100:.2f}%")
    print(f"Buy & Hold Return:     {market_return * 100:.2f}%")
    print(f"Sharpe Ratio:          {sharpe:.2f}")

    # Count how many trades we actually took
    n_trades = np.count_nonzero(signals)
    print(f"Total Trades Taken:    {n_trades} out of {len(signals)} days")

def main():
    print("--- USD/INR Prediction Pipeline ---")

    # 1. Data
    try:
        raw_df = data_loader.get_raw_data()
    except Exception as e:
        print(e)
        return

    # 2. Features
    features_df = feature_engg.create_features(raw_df)

    # 3. Modeling
    X_train, X_test, y_train, y_test = model_train.split_data(features_df)
    trained_models, _, _ = model_train.train_models(X_train, X_test, y_train, y_test)

    # 4. Backtesting (Using XGBoost)
    best_model = trained_models['XGBoost']
    backtest_strategy(best_model, X_test, y_test, features_df)

    print("\n--- Project Complete ---")


if __name__ == "__main__":
    main()