# ==========================
# Air Quality Prediction (PM2.5 Forecasting)
# ==========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from models.lstm_model import build_lstm_model

sns.set_style('whitegrid')

# --------------------------
# Step 1 — Load and Explore Dataset
# --------------------------

os.makedirs("results_aq", exist_ok=True)

# Load dataset
df = pd.read_csv('data/city_day.csv')
print("\n===== DATASET INFO =====")
print(df.head())
print(df.info())

# --------------------------
# Step 2 — Filter and Clean Data
# --------------------------

# Choose a single city for modeling, e.g., Delhi
city_name = "Delhi"
df_city = df[df['City'] == city_name].copy()

# Keep relevant columns
df_city = df_city[['Date', 'PM2.5', 'AQI']]

# Handle missing values
df_city = df_city.ffill().bfill()

# Convert Date column
df_city['Date'] = pd.to_datetime(df_city['Date'])
df_city.sort_values('Date', inplace=True)
df_city.set_index('Date', inplace=True)

print(f"\nData prepared for city: {city_name}")
print(df_city.describe())

# --------------------------
# Step 3 — Exploratory Data Analysis (EDA)
# --------------------------

plt.figure(figsize=(12,4))
sns.lineplot(data=df_city, x=df_city.index, y='PM2.5')
plt.title(f"{city_name} - PM2.5 Levels Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig("results_aq/pm25_trend.png", dpi=300)
plt.show()

plt.figure(figsize=(12,4))
sns.lineplot(data=df_city, x=df_city.index, y='AQI')
plt.title(f"{city_name} - AQI Levels Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("results_aq/aqi_trend.png", dpi=300)
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df_city.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (PM2.5 vs AQI)")
plt.tight_layout()
plt.savefig("results_aq/correlation_heatmap.png", dpi=300)
plt.show()

# --------------------------
# Step 4 — Preprocess for LSTM
# --------------------------

# Target variable: PM2.5
values = df_city['PM2.5'].values.reshape(-1,1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

TIME_STEPS = 10  # number of past days used to predict next day

def create_sequences(data, time_steps=TIME_STEPS):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTraining Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Shape:  {X_test.shape}, {y_test.shape}")

# --------------------------
# Step 5 — Build & Train LSTM Model
# --------------------------

model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

history = model.fit(
    X_train, y_train,
    epochs=20, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# --------------------------
# Step 6 — Evaluate Model
# --------------------------

y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"\nTest MSE for {city_name}: {mse:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='Actual', linewidth=2)
plt.plot(y_pred_inv, label='Predicted')
plt.title(f"{city_name} - Actual vs Predicted PM2.5 Levels")
plt.xlabel("Time Steps")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.tight_layout()
plt.savefig("results_aq/pred_vs_actual.png", dpi=300)
plt.show()

# Loss Curve
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("results_aq/loss_curve.png", dpi=300)
plt.show()

# --------------------------
# Step 7 — Manual Input Prediction
# --------------------------

print("\n====================== MANUAL INPUT TEST ======================")
print(f"This model predicts the next PM2.5 level based on the last {TIME_STEPS} days of readings.")

try:
    user_input = input(f"Enter {TIME_STEPS} recent PM2.5 values separated by commas:\n")
    if user_input.strip():
        vals = [float(v) for v in user_input.split(",")]
        if len(vals) == TIME_STEPS:
            seq = scaler.transform(np.array(vals).reshape(-1,1))
            seq = np.expand_dims(seq, axis=0)
            pred_next = model.predict(seq)
            pred_val = scaler.inverse_transform(pred_next)[0][0]
            print(f"\nPredicted Next PM2.5 Level: {pred_val:.3f} µg/m³")

            plt.figure(figsize=(8,4))
            plt.plot(range(1, TIME_STEPS+1), vals, marker='o', label="Last Days")
            plt.scatter(TIME_STEPS+1, pred_val, color='r', label="Prediction")
            plt.title(f"{city_name} - Manual Input Prediction")
            plt.xlabel("Days")
            plt.ylabel("PM2.5 (µg/m³)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("results_aq/manual_input_prediction.png", dpi=300)
            plt.show()
        else:
            print(f"⚠️ Please enter exactly {TIME_STEPS} values.")
except Exception as e:
    print("⚠️ Manual input error:", e)

# --------------------------
# Step 8 — Summary Table
# --------------------------

summary = pd.DataFrame({
    "City": [city_name],
    "Model": ["LSTM"],
    "Mean Squared Error (MSE)": [round(mse, 4)],
})
print("\n====================== SUMMARY TABLE ======================")
print(summary.to_string(index=False))
summary.to_csv("results_aq/model_summary.csv", index=False)
print("\n✅ Summary saved to results_aq/model_summary.csv")
