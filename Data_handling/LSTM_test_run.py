from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(69)


data_path = Path("selected_100_normalised.parquet")

max_min_path = Path("global_scaler.csv")

def load_data():
    df = pd.read_parquet(data_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

    scaler_df = pd.read_csv(max_min_path)
    global_kwh_min = float(scaler_df["global_kwh_min"].iloc[0])
    global_kwh_max = float(scaler_df["global_kwh_max"].iloc[0])

    global_temp_min = float(scaler_df["global_temp_min"].iloc[0])
    global_temp_max = float(scaler_df["global_temp_max"].iloc[0])

    global_hum_min = float(scaler_df["global_hum_min"].iloc[0])
    global_hum_max = float(scaler_df["global_hum_max"].iloc[0])

    return df, global_kwh_min, global_kwh_max, global_temp_min, global_temp_max, global_hum_min, global_hum_max

df, global_kwh_min, global_kwh_max, global_temp_min, global_temp_max, global_hum_min, global_hum_max = load_data()

print(df.head())
print(global_kwh_min)
print(global_kwh_max)

print(global_temp_min)
print(global_temp_max)

print(global_hum_min)
print(global_hum_max)

print("Unique houses:", df["LCLid"].nunique())
print("Shape:", df.shape)
print(df["split"].value_counts())


#Split data back into train test and val

feature_cols = [
    "kwh",
    "hour_sin", "hour_cos",
    "year_sin", "year_cos",
    "dow_sin", "dow_cos",
    "weekend", "temperature", "humidity"
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

#test on house index 1
unique_houses = df["LCLid"].unique()

#change to get unique house
house_id = unique_houses[0]
train_df, val_df, test_df = get_house_split(df, house_id)

print("House:", house_id)
print(train_df.head())
print(val_df.head())
print(test_df.head())
print(len(train_df), len(val_df), len(test_df))


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


X_train, y_train = make_xy(train_df, window_size=24, target_col="kwh")
X_val, y_val = make_xy(val_df, window_size=24, target_col="kwh")
X_test, y_test = make_xy(test_df, window_size=24, target_col="kwh")

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


#unscale function
def unscale(arr_scaled, min_val, max_val):
    return arr_scaled * (max_val - min_val) + min_val


#calculate evaluation metrics
def evaluate_predictions(y_scaled, pred_scaled, min_val, max_val):
    y_raw = unscale(y_scaled, min_val, max_val)
    pred_raw = unscale(pred_scaled, min_val, max_val)

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

model = build_lstm_2x20(X_train.shape[1:])
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    verbose=1,
    callbacks=[early_stop]
)

pred_scaled = model.predict(X_test, verbose=0).flatten()

metrics = evaluate_predictions(
    y_scaled=y_test,
    pred_scaled=pred_scaled,
    min_val=global_kwh_min,
    max_val=global_kwh_max
)
#check to see if overfitting
best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

train_loss_at_best_val = float(history.history["loss"][best_epoch - 1])
best_val_loss = float(history.history["val_loss"][best_epoch - 1])

final_train_loss = float(history.history["loss"][-1])
final_val_loss = float(history.history["val_loss"][-1])

epochs_run = len(history.history["loss"])
generalisation_gap = best_val_loss - train_loss_at_best_val

print("\nTraining diagnostics:")
print("Epochs run:", epochs_run)
print("Best epoch:", best_epoch)
print("Train loss at best val epoch:", train_loss_at_best_val)
print("Best validation loss:", best_val_loss)
print("Final train loss:", final_train_loss)
print("Final validation loss:", final_val_loss)
print("Generalisation gap at best epoch:", generalisation_gap)


print(metrics)

model.save("house_1_lstm_2x20.keras")