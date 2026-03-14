from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

import random

'''
Initial baseline for 2 layer LSTM with 20 units each (FL paper architecture)
Trained and evaluated on 100 randomly selected houses with good data coverage and cleaned data

Seeded run allowing for fair comparison with other models
'''

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(69)
np.random.seed(69)       # Ensures NumPy windowing/math is consistent
random.seed(69)          # Ensures Python's internal loops are consistent


data_path = Path("selected_100_households_hourly_scaled_with_splits.parquet")

max_min_path = Path("global_kwh_scaler.csv")

def load_data():
    df = pd.read_parquet(data_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

    scaler_df = pd.read_csv(max_min_path)
    global_kwh_min = float(scaler_df["global_kwh_min"].iloc[0])
    global_kwh_max = float(scaler_df["global_kwh_max"].iloc[0])

    return df, global_kwh_min, global_kwh_max

'''
df, global_kwh_min, global_kwh_max = load_data()

print(df.head())
print(global_kwh_min)
print(global_kwh_max)

print("Unique houses:", df["LCLid"].nunique())
print("Shape:", df.shape)
print(df["split"].value_counts())
'''

#Split data back into train test and val

feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend",
]

#define the model input features

def get_house_split(df: pd.DataFrame, house_id: str):
    house_df = df[df["LCLid"] == house_id].copy().sort_values("DateTime")   #sort by date and time and obtain values for individual house

    train_df = house_df[house_df["split"] == "train"].copy()    #get training data
    val_df = house_df[house_df["split"] == "val"].copy()
    test_df = house_df[house_df["split"] == "test"].copy()

    train_df = train_df[feature_cols].copy()        #organise columns by input features as defined above
    val_df = val_df[feature_cols].copy()
    test_df = test_df[feature_cols].copy()

    return train_df, val_df, test_df
'''
house_id = df["LCLid"].iloc[0]
train_df, val_df, test_df = get_house_split(df, house_id)

print("House:", house_id)
print(train_df.head())
print(val_df.head())
print(test_df.head())
print(len(train_df), len(val_df), len(test_df))
'''

#create rolling windows for training
def make_xy(df_house: pd.DataFrame, window_size: int = 24, target_col: str = "kwh"):
    values = df_house.to_numpy(dtype=np.float32)
    target_idx = df_house.columns.get_loc(target_col)       #get index of column

    X = []
    y = []

    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size, target_idx])

    return np.array(X), np.array(y)

'''
X_train, y_train = make_xy(train_df, window_size=24, target_col="kwh")
X_val, y_val = make_xy(val_df, window_size=24, target_col="kwh")
X_test, y_test = make_xy(test_df, window_size=24, target_col="kwh")

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
'''

#unscale function
def unscale_kwh(arr_scaled, min_val, max_val):
    return arr_scaled * (max_val - min_val) + min_val


#calculate evaluation metrics
def evaluate_predictions(y_scaled, pred_scaled, min_val, max_val):
    y_raw = unscale_kwh(y_scaled, min_val, max_val)
    pred_raw = unscale_kwh(pred_scaled, min_val, max_val)

    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    mae = mean_absolute_error(y_raw, pred_raw)

    y_std = np.std(y_raw)
    y_mean = np.mean(y_raw)

    if y_std != 0:
        nrmse_std = rmse / y_std
    else:
        nrmse_std = np.nan

    if y_mean != 0:
        nrmse_mean = rmse / y_mean

    else:
        nrmse_mean = np.nan


    return {
        "rmse_kwh": rmse,
        "mae_kwh": mae,
        "nrmse_std": nrmse_std,
        "nrmse_mean": nrmse_mean,
    }


#build model and compiler
def build_lstm_2x20(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(20, return_sequences=True),
        LSTM(20),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model





def train_one_lstm(X_train, y_train, X_val, y_val):
    model = build_lstm_2x20(X_train.shape[1:])


    es = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        verbose=0,
        callbacks=[es]
    )

    return model

WINDOW_SIZE = 24
TARGET_COL = "kwh"


df, global_kwh_min, global_kwh_max = load_data()    #load data
#house_ids = sorted(df["LCLid"].unique())[:5]
house_ids = sorted(df["LCLid"].unique())
results = []

for i, house_id in enumerate(house_ids, start=1):
    print(f"Processing {i}/{len(house_ids)}: {house_id}")

    train_df, val_df, test_df = get_house_split(df, house_id)

    X_train, y_train = make_xy(train_df, window_size=WINDOW_SIZE, target_col=TARGET_COL)
    X_val, y_val = make_xy(val_df, window_size=WINDOW_SIZE, target_col=TARGET_COL)
    X_test, y_test = make_xy(test_df, window_size=WINDOW_SIZE, target_col=TARGET_COL)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f"Skipping {house_id}: insufficient samples after windowing")
        continue

    tf.keras.backend.clear_session()

    model = train_one_lstm(X_train, y_train, X_val, y_val)
    pred_scaled = model.predict(X_test, verbose=0).flatten()

    metrics = evaluate_predictions(
        y_scaled=y_test,
        pred_scaled=pred_scaled,
        min_val=global_kwh_min,
        max_val=global_kwh_max
    )

    results.append({
        "house_id": house_id,
        "rmse_kwh": metrics["rmse_kwh"],
        "mae_kwh": metrics["mae_kwh"],
        "nrmse_std": metrics["nrmse_std"],
        "nrmse_mean": metrics["nrmse_mean"],
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
    })

    print(f"[{i}/{len(house_ids)}] Finished {house_id} | Test RMSE: {metrics['rmse_kwh']:.6f}")

results_df = pd.DataFrame(results)
if results_df.empty:
    raise ValueError("No houses were evaluated. Check windowing or splits.")

print("\nPer-house LSTM results:")
print(results_df.head())

print("\nLSTM summary:")
print("Mean RMSE:", results_df["rmse_kwh"].mean())
print("Median RMSE:", results_df["rmse_kwh"].median())
print("Std RMSE:", results_df["rmse_kwh"].std())
print("Mean MAE:", results_df["mae_kwh"].mean())
print("Mean NRMSE/std:", results_df["nrmse_std"].mean())
print("Mean NRMSE/mean:", results_df["nrmse_mean"].mean())
print("Houses evaluated:", results_df["house_id"].nunique())

results_df.to_csv("lstm_per_house_results.csv", index=False)


#summary statistics
summary_df = pd.DataFrame([{
    "model": "lstm_2x20",
    "mean_rmse_kwh": results_df["rmse_kwh"].mean(),
    "median_rmse_kwh": results_df["rmse_kwh"].median(),
    "std_rmse_kwh": results_df["rmse_kwh"].std(),
    "mean_mae_kwh": results_df["mae_kwh"].mean(),
    "mean_nrmse_std": results_df["nrmse_std"].mean(),
    "mean_nrmse_mean": results_df["nrmse_mean"].mean(),
    "n_houses": results_df["house_id"].nunique()
}])

summary_df.to_csv("lstm_summary.csv", index=False)
print(summary_df)