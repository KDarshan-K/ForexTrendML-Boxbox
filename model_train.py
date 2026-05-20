# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import config


def split_data(df):
    """Chronological train/test split."""
    X = df.drop(columns=['target_direction', config.PRICE_COLUMN, 'returns'])
    y = df['target_direction']

    # Split chronologically (Shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Trains bake-off models."""

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'LogReg': LogisticRegression(),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}

    print("\n--- Model Bake-off Results ---")
    for name, model in models.items():
        if name == 'LogReg':
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = model
        print(f"{name} Accuracy: {acc:.4f}")

        # Show Feature Importance for XGBoost
        if name == 'XGBoost':
            imps = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            print("\n[XGBoost Top 5 Features]:")
            print(imps.head(5))

    return results, X_test, y_test