import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error as mse

from tensorflow.keras.layers import Flatten
from sklearn.metrics import mean_absolute_error

def standardise(array, mean, std):
    array = (array - mean) / std
    return array

def unstandardise(array, mean, std):
    array = (array*std) + mean
    return array

def x_y(df, window_size=12, target_col='kwh'):
    df_as_np = df.to_numpy(dtype=np.float32)
    target_idx = df.columns.get_loc(target_col) #get index of kwh column

    x = []
    y = []

    for i in range(len(df_as_np) - window_size):
        x.append(df_as_np[i:i + window_size])              # past 24 hours, all features
        y.append(df_as_np[i + window_size, target_idx])   # next hour demand only for now

    return np.array(x), np.array(y)

def plot_predictions(model, x, y, start = 0, end = 1000):

    predictions = model.predict(x).flatten()
    predictions = unstandardise(predictions, demand_mean, demand_std)

    y_actuals = unstandardise(y, demand_mean, demand_std)
    df = pd.DataFrame(data={'prediction': predictions, 'target': y_actuals})
    #plot
    plt.plot(df['prediction'][start:end], label='Preds')
    plt.plot(df['target'][start:end], label='Actuals')

    plt.legend()
    plt.show()

    return df, mse(predictions, y_actuals)

'''
Extract relevant data 
'''

df=pd.read_csv('Partitioned LCL Data/Small LCL Data/LCL-June2015v2_0.csv')

selected_house = df[df['LCLid'] == 'MAC000018'].copy() #only take houses with MAC000002 ID into new dataframe

print(selected_house.head())

selected_house = selected_house.rename(columns={'KWH/hh (per half hour) ': 'kwh'})

#remove any duplicate rows
selected_house = selected_house.drop_duplicates(
    subset=['LCLid', 'stdorToU', 'DateTime', 'kwh'],
    keep='first'
)

selected_house['DateTime'] = pd.to_datetime(selected_house['DateTime']) #set it to date time object
selected_house = selected_house.set_index('DateTime')   #set index as datatime

print(selected_house.head())
print(selected_house.shape)

print(selected_house[:24])

#convert data to numbers (originally strings)
selected_house['kwh'] = pd.to_numeric(selected_house['kwh'], errors='coerce')

selected_house.index = selected_house.index - pd.Timedelta(minutes=30)  #shift data back by 30mins to allow summation to sum correctly
hourly = pd.DataFrame(selected_house['kwh'].resample('h').sum()) #sum hourly


#print(hourly)
#print(hourly.shape)


'''
Add in day sin cos and year sin cos for model input features
'''

day_duration = 60*60*24
year_duration = day_duration*365.2425

seconds = hourly.index.map(pd.Timestamp.timestamp)


print(hourly)
#compute hourly sin and cos values

hourly['Hour sin'] = np.sin(seconds* 2 *np.pi/day_duration)
hourly['Hour cos'] = np.cos(seconds* 2 *np.pi/day_duration)

hourly['Year sin'] = np.sin(seconds* 2 *np.pi/year_duration)
hourly['Year cos'] = np.cos(seconds* 2 *np.pi/year_duration)

# 7-day cycle
dow = hourly.index.dayofweek
hourly['dow_sin'] = np.sin(2 * np.pi * dow / 7)
hourly['dow_cos'] = np.cos(2 * np.pi * dow / 7)

#weekend flag
hourly['weekend'] = (hourly.index.dayofweek >= 5).astype(int)

print(hourly)


#split into training, validation and test set
total_examples = len(hourly)
t_s = round(total_examples*0.6)
v_s = round(total_examples*0.2)

#split into training, test and validation sets
train_set = hourly.iloc[:t_s].copy()
val_set = hourly.iloc[t_s: v_s + t_s].copy()
test_set = hourly.iloc[v_s + t_s:].copy()

#get mean and std from train set
demand_mean = train_set['kwh'].mean()
demand_std = train_set['kwh'].std()

#standardise data
train_set['kwh'] = standardise(train_set['kwh'], demand_mean, demand_std)
val_set['kwh'] = standardise(val_set['kwh'], demand_mean, demand_std)
test_set['kwh'] = standardise(test_set['kwh'], demand_mean, demand_std)



x_train, y_train = x_y(train_set)
x_val, y_val = x_y(val_set)
x_test, y_test = x_y(test_set)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)





'''
model = Sequential([

    Input(shape=(x_train.shape[1:])),
    LSTM(64),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'linear')

])
'''
#now train model
model = Sequential([
    Input(shape=(x_train.shape[1:])),
    LSTM(20, return_sequences=True),
    LSTM(20),
    Dense(1, activation = 'linear')
])
model.summary()

model.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])

#The Early Stopper: Stop if we don't improve for 5 hours (epochs)
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

cp = ModelCheckpoint('model_checkpoint.keras', save_best_only=True)


model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[cp, es] #only saves model (checkpoint) if it performs the best so far, and stops training if MSE doesn't decrease after 5 rounds
)


model = load_model('model_checkpoint.keras')

#run inference using test set and plot
_, test_mse = plot_predictions(model, x_test, y_test)

print(f"rmse: {np.sqrt(test_mse)}")





def evaluate_scaled_predictions(y_scaled, pred_scaled, mean, std, label="Model"):
    y_raw = unstandardise(y_scaled, mean, std)
    pred_raw = unstandardise(pred_scaled, mean, std)

    rmse = np.sqrt(mse(y_raw, pred_raw))
    mae = mean_absolute_error(y_raw, pred_raw)

    y_std = np.std(y_raw)
    y_mean = np.mean(y_raw)

    nrmse_std = rmse / y_std if y_std != 0 else np.nan
    nrmse_mean = rmse / y_mean if y_mean != 0 else np.nan

    print(f"\n--- {label} ---")
    print(f"RMSE (kWh): {rmse:.4f}")
    print(f"MAE  (kWh): {mae:.4f}")
    print(f"NRMSE / std:  {nrmse_std:.4f}")
    print(f"NRMSE / mean: {nrmse_mean:.4f}")

    return {
        "label": label,
        "rmse": rmse,
        "mae": mae,
        "nrmse_std": nrmse_std,
        "nrmse_mean": nrmse_mean,
        "y_raw": y_raw,
        "pred_raw": pred_raw
    }

def print_skill_against_persistence(model_results, persistence_results):
    skill_rmse = 1 - (model_results["rmse"] / persistence_results["rmse"])
    skill_mae = 1 - (model_results["mae"] / persistence_results["mae"])

    print(f"\n{model_results['label']} vs Persistence")
    print(f"RMSE skill: {skill_rmse:.4f}")
    print(f"MAE  skill: {skill_mae:.4f}")

def build_dense_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=[RootMeanSquaredError()]
    )
    return model

# ------------------------
# Persistence baseline
# ------------------------
target_idx = train_set.columns.get_loc('kwh')
persistence_pred_scaled = x_test[:, -1, target_idx]

persistence_results = evaluate_scaled_predictions(
    y_scaled=y_test,
    pred_scaled=persistence_pred_scaled,
    mean=demand_mean,
    std=demand_std,
    label="Persistence"
)

# ------------------------
# Dense baseline
# ------------------------
dense_model = build_dense_model(x_train.shape[1:])

dense_cp = ModelCheckpoint('dense_checkpoint.keras', save_best_only=True)
dense_es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

dense_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[dense_cp, dense_es]
)

dense_model = load_model('dense_checkpoint.keras')
dense_pred_scaled = dense_model.predict(x_test, verbose=0).flatten()

dense_results = evaluate_scaled_predictions(
    y_scaled=y_test,
    pred_scaled=dense_pred_scaled,
    mean=demand_mean,
    std=demand_std,
    label="Dense"
)

# ------------------------
# LSTM results
# ------------------------
lstm_model = load_model('model_checkpoint.keras')
lstm_pred_scaled = lstm_model.predict(x_test, verbose=0).flatten()

lstm_results = evaluate_scaled_predictions(
    y_scaled=y_test,
    pred_scaled=lstm_pred_scaled,
    mean=demand_mean,
    std=demand_std,
    label="LSTM"
)

# ------------------------
# Skill vs persistence
# ------------------------
print_skill_against_persistence(dense_results, persistence_results)
print_skill_against_persistence(lstm_results, persistence_results)

# ------------------------
# Summary table
# ------------------------
summary_df = pd.DataFrame([
    {
        "Model": persistence_results["label"],
        "RMSE_kWh": persistence_results["rmse"],
        "MAE_kWh": persistence_results["mae"],
        "NRMSE_std": persistence_results["nrmse_std"],
        "NRMSE_mean": persistence_results["nrmse_mean"]
    },
    {
        "Model": dense_results["label"],
        "RMSE_kWh": dense_results["rmse"],
        "MAE_kWh": dense_results["mae"],
        "NRMSE_std": dense_results["nrmse_std"],
        "NRMSE_mean": dense_results["nrmse_mean"]
    },
    {
        "Model": lstm_results["label"],
        "RMSE_kWh": lstm_results["rmse"],
        "MAE_kWh": lstm_results["mae"],
        "NRMSE_std": lstm_results["nrmse_std"],
        "NRMSE_mean": lstm_results["nrmse_mean"]
    }
])

print(summary_df)
