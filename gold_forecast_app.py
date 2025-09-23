import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import streamlit as st

# Page configuration
st.set_page_config(page_title="Gold Price Forecast", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Upload a CSV with columns **Date** and **GLD** to begin.")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown("Created by Yash Nalavade")
    st.markdown("           Abhay Singh")
    st.markdown("           Abhay Navale")

# --- Main Title ---
st.markdown("""
    <h1 style="text-align:center; color:goldenrod;">🪙 Gold Price Forecasting</h1>
    <p style="text-align:center;">Powered by Random Forest Regression | 365-Day Projection</p>
    <hr>
""", unsafe_allow_html=True)

# --- Main App Logic ---
if uploaded_file:
    # Load and preprocess
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')[['Date', 'GLD']].copy()
    df = df.set_index('Date')

    # Create lag features
    for i in range(1, 8):
        df[f'lag_{i}'] = df['GLD'].shift(i)
    df.dropna(inplace=True)

    X = df.drop(columns=['GLD'])
    y = df['GLD']
    X_train, X_test = X[:-365], X[-365:]
    y_train, y_test = y[:-365], y[-365:]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Display evaluation
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("📉 Test MSE", f"{mse:.2f}")
    with col2:
        st.success("Model trained and test predictions complete!")

    # Forecasting
    last_known = df.iloc[-7:].copy()
    future_preds = []
    current_date = df.index[-1]

    for _ in range(365):
        input_data = [last_known['GLD'].iloc[-j] for j in range(1, 8)]
        input_df = pd.DataFrame([input_data], columns=[f'lag_{i}' for i in range(1, 8)])
        pred = model.predict(input_df)[0]
        future_preds.append((current_date + timedelta(days=1), pred))
        new_row = pd.DataFrame([[pred]], columns=['GLD'], index=[current_date + timedelta(days=1)])
        last_known = pd.concat([last_known, new_row])
        last_known = last_known[-7:]
        current_date += timedelta(days=1)

    # --- Visualization ---
    st.subheader("📈 Gold Price Forecast for Next 365 Days")

    dates = [d for d, _ in future_preds]
    prices = [p for _, p in future_preds]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, label='Predicted Price', color='orange', linewidth=2.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title('Gold Price Forecast (365 Days Ahead)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Forecast Data Table
    st.subheader("🔢 Forecasted Values")
    forecast_df = pd.DataFrame(future_preds, columns=["Date", "Predicted GLD"])
    st.dataframe(forecast_df.set_index("Date").style.format({"Predicted GLD": "{:.2f}"}))
else:
    st.warning("👆 Upload a CSV file on the sidebar to get started.")
