# Gold-Price-Prediction-using-ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import matplotlib.dates as mdates

# Load the dataset
df = pd.read_csv("gold_price_data.csv")

# Parse date and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Use 'GLD' as the target (Gold ETF price)
df = df[['Date', 'GLD']].copy()
df = df.set_index('Date')

# Create lag features
for i in range(1, 8):  # 7 days history
    df[f'lag_{i}'] = df['GLD'].shift(i)

# Drop rows with NaN values (from lagging)
df.dropna(inplace=True)

# Split into features and target
X = df.drop(columns=['GLD'])
y = df['GLD']

# Train/test split (last 365 days for testing)
X_train, X_test = X[:-365], X[-365:]
y_train, y_test = y[:-365], y[-365:]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.2f}")

# Forecast next 365 days
last_known = df.iloc[-7:].copy()
future_preds = []
current_date = df.index[-1]

for i in range(365):
    # Prepare input features for prediction
    input_data = [last_known['GLD'].iloc[-j] for j in range(1, 8)]
    input_df = pd.DataFrame([input_data], columns=[f'lag_{i}' for i in range(1, 8)])
    
    # Predict next day
    pred = model.predict(input_df)[0]
    future_preds.append((current_date + timedelta(days=1), pred))

    # Update last_known with new prediction
    new_row = pd.DataFrame([[pred]], columns=['GLD'], index=[current_date + timedelta(days=1)])
    last_known = pd.concat([last_known, new_row])
    last_known = last_known[-7:]  # Keep only last 7 for lagging
    current_date += timedelta(days=1)

# Display future predictions
print("\n📅 Predicted Gold Prices for the Next 365 Days:")
for date, price in future_preds:
    print(f"{date.date()}: ${price:.2f}")

# Optional: Plot future predictions
# Plotting future predictions
dates = [d for d, _ in future_preds]
prices = [p for _, p in future_preds]

plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='Predicted Price', color='gold')

# X-axis: ticks every 2 years, format as year
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title('Gold Price Forecast for the Next 365 Days')
plt.xlabel('Year')
plt.ylabel('Gold Price ($)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # 👈 makes x-axis labels readable
plt.tight_layout()
plt.show()
