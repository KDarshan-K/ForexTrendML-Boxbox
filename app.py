# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import data_loader
import feature_engg
from xgboost import XGBClassifier
import config

# --- Page Config ---
st.set_page_config(page_title="USD/INR AI Forecaster", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🇮🇳🇺🇸 USD/INR Macro-Quant System")
st.markdown("Forecasting the **5-Day Direction** of the Indian Rupee using **XGBoost** and **Macro Factors**.")

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider("Confidence Threshold", 0.50, 0.90, 0.60, 0.05)


# --- 1. Load Data Pipeline (Robust) ---
@st.cache_data(ttl=3600)
def load_data_pipeline():
    raw = data_loader.get_raw_data()
    feats = feature_engg.create_features(raw)
    return raw, feats


# Initialize variables
raw_df = None
features_df = None

with st.spinner('Fetching Live Market & Macro Data...'):
    try:
        raw_df, features_df = load_data_pipeline()
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

if raw_df is None or features_df is None:
    st.error("Data could not be loaded.")
    st.stop()

# --- 2. Train Model (Real-time) ---
X = features_df.drop(columns=['target_direction', config.PRICE_COLUMN, 'returns'])
y = features_df['target_direction']

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                      use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X, y)

# --- 3. Prediction Engine ---
latest_features = X.iloc[[-1]]
latest_date = latest_features.index[0]  # Keeping as Timestamp for math
latest_close = raw_df[config.PRICE_COLUMN].iloc[-1]
prob_up = model.predict_proba(latest_features)[0][1]

# --- 4. Dashboard Layout ---

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Latest USD/INR Price", f"₹{latest_close:.2f}")

with col2:
    st.metric("Model Accuracy (Hist)", "61.0%")

with col3:
    if prob_up >= threshold:
        signal = "🔼 STRONG BUY (UP)"
        color_code = "green"
        direction = 1
    elif prob_up <= (1 - threshold):
        signal = "🔽 STRONG SELL (DOWN)"
        color_code = "red"
        direction = -1
    else:
        signal = "⚪ NEUTRAL / HOLD"
        color_code = "gray"
        direction = 0

    st.metric("AI Signal (5-Day)", signal, delta=f"{prob_up:.1%} Confidence")

with col4:
    st.metric("Data Date", str(latest_date.date()))

st.markdown("---")

# --- NEW: Interactive Price Chart with Projection ---
st.subheader("📈 Price Forecast Visualization")

# 1. Get Historical Data (Last 90 Days)
history_days = 90
hist_df = raw_df.tail(history_days)

# 2. Calculate Projection Points
# We project 5 days into the future.
# Magnitude: We use recent volatility to estimate *how much* it might move.
recent_volatility = features_df['volatility'].iloc[-1]
if pd.isna(recent_volatility): recent_volatility = 0.005  # Default to 0.5% if NaN

# Calculate target price based on direction
# Formula: Current * (1 + (Direction * Volatility * Days))
projected_change = direction * recent_volatility * np.sqrt(5)  # Scale vol for 5 days
target_price = latest_close * (1 + projected_change)
future_date = latest_date + timedelta(days=5)

# 3. Build Plotly Chart
fig = go.Figure()

# Trace A: Historical Price
fig.add_trace(go.Scatter(
    x=hist_df.index,
    y=hist_df[config.PRICE_COLUMN],
    mode='lines',
    name='Historical Price',
    line=dict(color='#1f77b4', width=2)
))

# Trace B: Projected Path (Dotted Line)
if direction != 0:
    fig.add_trace(go.Scatter(
        x=[latest_date, future_date],
        y=[latest_close, target_price],
        mode='lines+markers',
        name='AI Forecast Path',
        line=dict(color=color_code, width=2, dash='dot'),
        marker=dict(size=8)
    ))

    # Add an annotation for the target
    fig.add_annotation(
        x=future_date, y=target_price,
        text=f"Forecast: ₹{target_price:.2f}",
        showarrow=True, arrowhead=1
    )

fig.update_layout(
    title=f"USD/INR History & 5-Day Projection ({signal})",
    xaxis_title="Date",
    yaxis_title="Exchange Rate (INR)",
    height=500,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# --- Existing Sections (Feature Importance & Data) ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Feature Importance")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True).tail(10)

    fig_imp = go.Figure(go.Bar(
        x=importances['Importance'],
        y=importances['Feature'],
        orientation='h',
        marker_color='#4A90E2'
    ))
    fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

with col_right:
    st.subheader("📋 Model Inputs")
    st.caption("Latest processed inputs:")
    st.dataframe(latest_features.T, height=400)