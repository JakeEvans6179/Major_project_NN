from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

"""
Loading saved model and running inference (prediction)

Need to load in the dataframe with the household data, extract test data set from dataframe
Perform windowing (organise data into format suitable for LSTM)
Run predictions
"""

#load in demand data (normalised)
def load_data():
    df = pd.read_parquet("selected_100_normalised.parquet")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

    scaler_df = pd.read_csv("global_scaler.csv")
    global_kwh_min = float(scaler_df["global_kwh_min"].iloc[0])
    global_kwh_max = float(scaler_df["global_kwh_max"].iloc[0])

    global_temp_min = float(scaler_df["global_temp_min"].iloc[0])
    global_temp_max = float(scaler_df["global_temp_max"].iloc[0])

    global_hum_min = float(scaler_df["global_hum_min"].iloc[0])
    global_hum_max = float(scaler_df["global_hum_max"].iloc[0])

    return df, global_kwh_min, global_kwh_max, global_temp_min, global_temp_max, global_hum_min, global_hum_max

df, global_kwh_min, global_kwh_max, global_temp_min, global_temp_max, global_hum_min, global_hum_max = load_data()

#split into format suitable to be fed into lstm
def make_xy(df_house: pd.DataFrame, window_size: int = 24, target_col: str = "kwh"):
    values = df_house.to_numpy(dtype=np.float32)
    target_idx = df_house.columns.get_loc(target_col)       #get index of column

    X = []
    y = []

    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size, target_idx])

    return np.array(X), np.array(y)


feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
]

def get_house_split(df: pd.DataFrame, house_id: str):
    house_df = df[df["LCLid"] == house_id].copy().sort_values("DateTime")   #sort by date and time and obtain values for individual house

    test_df = house_df[house_df["split"] == "test"].copy()  #inference only, so only need test data


    test_df = test_df[feature_cols].copy()

    return test_df

#test on house index 1
unique_houses = df["LCLid"].unique()


house_id = unique_houses[0]
test_df = get_house_split(df, house_id)

print("House:", house_id)

print(test_df.head())
print(len(test_df))

x_test, y_test = make_xy(test_df, window_size=24, target_col="kwh")

# Define unscale function ONCE
def unscale(arr_scaled, min_val, max_val):
    return arr_scaled * (max_val - min_val) + min_val

# Calculate evaluation metrics
def evaluate_predictions(y_scaled, pred_scaled, min_val, max_val):
    y_raw = unscale(y_scaled, min_val, max_val)
    pred_raw = unscale(pred_scaled, min_val, max_val)

    pred_raw = np.clip(pred_raw, a_min=0.0, a_max=None)

    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    mae = mean_absolute_error(y_raw, pred_raw)

    y_std = np.std(y_raw)
    y_mean = np.mean(y_raw)

    nrmse_std = rmse / y_std if y_std != 0 else np.nan
    nrmse_mean = rmse / y_mean if y_mean != 0 else np.nan

    return {
        "rmse_kwh": rmse,
        "mae_kwh": mae,
        "nrmse_std": nrmse_std,
        "nrmse_mean": nrmse_mean,
    }, y_raw, pred_raw


# Load Model
model = load_model("house_1_lstm_2x20.keras")

# Run predictions ONCE
print("Running predictions...")
pred_scaled = model.predict(x_test, verbose=0).flatten()




# Evaluate the predictions
metrics, y_raw, pred_raw = evaluate_predictions(
    y_scaled=y_test,
    pred_scaled=pred_scaled,
    min_val=global_kwh_min,
    max_val=global_kwh_max
)

print(f"\n--- Metrics for {house_id} ---")
print(f"RMSE (kWh): {metrics['rmse_kwh']:.4f}")
print(f"MAE (kWh):  {metrics['mae_kwh']:.4f}")

# Plotting (FIXED)
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Plot only a slice (e.g., first 336 hours = 2 weeks) so it's not a giant unreadable blob
slice_size = 336

# Make sure to plot the RAW (unscaled) actuals vs the RAW (unscaled) predictions
plt.plot(y_raw[:slice_size], label="Actual Demand", color="blue", alpha=0.6)
plt.plot(pred_raw[:slice_size], label="LSTM Prediction", color="red", linestyle="--", alpha=0.9)

plt.title(f"2-Week Demand Forecast vs Actuals ({house_id})")
plt.ylabel("Electricity Demand (kWh)")
plt.xlabel("Hours into Test Set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


