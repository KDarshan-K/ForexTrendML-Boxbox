# generate_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import data_loader
import feature_engg
import config

# --- Configuration ---
IMG_FILENAME = "prediction_graph.png"
CM_FILENAME = "confusion_matrix.png"

# 1. Load Data
print("Loading data...")
raw_df = data_loader.get_raw_data()
df = feature_engg.create_features(raw_df)

# 2. Prepare Train/Test Data (Chronological Split)
# We plot the results for the TEST period (the "unseen" future)
test_size = 0.2
split_idx = int(len(df) * (1 - test_size))

X = df.drop(columns=['target_direction', config.PRICE_COLUMN, 'returns'])
y = df['target_direction']

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Get the corresponding Price Data for the test set
test_dates = X_test.index
test_prices = raw_df.loc[test_dates, config.PRICE_COLUMN]

# 3. Train Model
print("Training model...")
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                      use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 4. Generate Predictions
# Use the same threshold logic as your report (0.60)
threshold = 0.60
proba = model.predict_proba(X_test)[:, 1]

# Create Signals
# Buy (1) if prob > 0.60, Sell (-1) if prob < 0.40
signals = np.where(proba > threshold, 1,
                   np.where(proba < (1 - threshold), -1, 0))

# ---------------------------------------------------------
# GRAPH 1: Price History with Prediction Signals (The "Results" Graph)
# ---------------------------------------------------------
print(f"Generating {IMG_FILENAME}...")

plt.figure(figsize=(12, 6))

# Plot the Actual Price
plt.plot(test_prices.index, test_prices, label='Actual USD/INR Price', color='black', alpha=0.6, linewidth=1)

# Overlay "Buy" Signals (Green Up Triangles)
buy_indices = np.where(signals == 1)[0]
if len(buy_indices) > 0:
    plt.scatter(test_prices.index[buy_indices], test_prices.iloc[buy_indices],
                marker='^', color='green', label='AI Predicted UP', s=100, zorder=5)

# Overlay "Sell" Signals (Red Down Triangles)
sell_indices = np.where(signals == -1)[0]
if len(sell_indices) > 0:
    plt.scatter(test_prices.index[sell_indices], test_prices.iloc[sell_indices],
                marker='v', color='red', label='AI Predicted DOWN', s=100, zorder=5)

# Formatting
plt.title(f'Model Prediction Performance (Test Data: {test_dates[0].date()} to {test_dates[-1].date()})', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate (INR)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Save
plt.tight_layout()
plt.savefig(IMG_FILENAME, dpi=300) # High resolution for Word/PDF
print("Graph saved!")

# ---------------------------------------------------------
# GRAPH 2: Confusion Matrix (Optional but great for Analysis)
# ---------------------------------------------------------
print(f"Generating {CM_FILENAME}...")

# Convert signals to binary for confusion matrix (ignoring neutrals for pure accuracy check)
# Here we just look at raw directional accuracy (Up vs Down) for the matrix
preds_binary = (proba > 0.5).astype(int)
cm = confusion_matrix(y_test, preds_binary)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up'])
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(CM_FILENAME, dpi=300)
print("Confusion Matrix saved!")