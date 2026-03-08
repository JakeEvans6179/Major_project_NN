import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error as mse

current_dir = os.getcwd()  # get current directory
print(f"Current directory: {current_dir}")

extracted_folder_path = tf.keras.utils.get_file(
    fname='jena_climate_2009_2016.csv.zip',  # Just the filename now!
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    cache_dir=current_dir,  # Tell it to use your current folder
    cache_subdir='.',  # Tell it NOT to create a 'datasets' subfolder
    extract=True,  # unzip file
)

# Join the folder path with the actual name of the CSV file inside it
csv_path = os.path.join(extracted_folder_path, "jena_climate_2009_2016.csv")
print(f"Dataset successfully saved to: {csv_path}")

df = pd.read_csv(csv_path)
# Tell Pandas to show all columns
pd.set_option('display.max_columns', None) #none means all the columns

print(df.head())  # .head() ensures you still only print the top 5 rows

df = df[5::6]  # start at 5th row, take every 6th row (ie every hour) [start : stop: step] --> no stop = go to end of dataset
# print(df.head())
print(df)

df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')  # converts datetime text on dataset to Datetime object, and sets our label index to these objects
print(df[:26])  # can still call the indexes using row number even though date time is the official label

# just doing univariate LSTM so just get temp
temp = df['T (degC)']
print(temp.head())  # now is a pandas series

#temp.plot()

#plt.show()




def plot_predictions(model, x, y, start = 0, end = 100):

    predictions = model.predict(x).flatten()
    predictions = unstandardise(predictions, temp_mean, temp_std)
    y = unstandardise(y, temp_mean, temp_std)
    df = pd.DataFrame(data={'prediction': predictions, 'target': y})
    plt.plot(df['prediction'][start:end], label='Preds')
    plt.plot(df['target'][start:end], label='Actuals')
    plt.legend()
    plt.show()


    return df, mse(y, predictions)



#compute seconds of day to use as input feature
temp_df = pd.DataFrame({'Temperature': temp})

#temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp) #maps index (time) to total seconds from 1970
elapsed_time = temp_df.index - temp_df.index[0]

#create new column and assign it the total seconds
temp_df['Seconds'] = elapsed_time.total_seconds()

#print(temp_df[:1])     #df['x'] looks for column x, df[0:y] looks for rows between index 0 and index y
print(temp_df.head())

#find number of seconds per day and year
day = 60*60*24
year = 365.2425*day     #used to compute day sine/cos and year sine/cos

temp_df['Day sin'] = np.sin(temp_df['Seconds'] * 2 * np.pi / day)
temp_df['Day cos'] = np.cos(temp_df['Seconds'] * 2 * np.pi / day)

temp_df['Year sin'] = np.sin(temp_df['Seconds'] * 2 * np.pi / year)
temp_df['Year cos'] = np.cos(temp_df['Seconds'] * 2 * np.pi / year)
#print(temp_df.head())
#now remove seconds column
temp_df = temp_df.drop(columns=['Seconds'])

print(temp_df.head())


#now need to preprocess data into same format as for univariate
'''
[[t1, ds1], [t2, ds2], [t3, ds3], [t4, ds4], [t5, ds5]] --> [t6]
[[t2, ds2], [t3, ds3], [t4, ds4], [t5, ds5], [t6, ds6]] --> [t7]

etc ^ not just ds (day sine) but dc, ys, yc
'''
# create function to organise variables
def df_to_x_y(df, window_size=24):
    df_as_np = df.to_numpy()  # convert to numpy array
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [r for r in df_as_np[i:i + window_size]]  #removed list around r, now that data is matrix, each r takes the row of matrix (is already a list) --> not like univariate
        X.append(row)

        label = df_as_np[i + window_size][0]  #takes the temp that the values are predicting
        y.append(label)

    return np.array(X), np.array(y)

def standardise(array, mean, std):
    array = (array - mean) / std
    return array

def unstandardise(array, mean, std):
    array = array*std + mean
    return array


train_df = temp_df.iloc[:60000].copy()      #split dataframe first
val_df   = temp_df.iloc[60000:65000].copy()
test_df  = temp_df.iloc[65000:].copy()

#calculate mean and std from training set only
temp_mean = train_df['Temperature'].mean()
temp_std = train_df['Temperature'].std()

#standardise separately --> since we window after this, standardising will do for both x and y
train_df['Temperature'] = standardise(train_df['Temperature'], temp_mean, temp_std)
val_df['Temperature'] = standardise(val_df['Temperature'], temp_mean, temp_std)
test_df['Temperature'] = standardise(test_df['Temperature'], temp_mean, temp_std)

#make the window structure for training
x_train, y_train = df_to_x_y(train_df)
x_val, y_val = df_to_x_y(val_df)
x_test, y_test = df_to_x_y(test_df)


model2 = load_model('model2_checkpoint.keras')


#run inference using test set and plot
_, model_mse = plot_predictions(model2, x_test, y_test)


#try persistence model t1, t2, t3, t4, t5, t6 --> t6
#basically just grab all the values from window size

y_persistence = x_test[:, -1, 0]    #selects all examples, takes last list in each 6 hour window, and selects index 0 (temperature)
y_persistence = unstandardise(y_persistence, temp_mean, temp_std)
y_test = unstandardise(y_test, temp_mean, temp_std)

persistence_mse = mse(y_test, y_persistence)

plt.plot(y_persistence[0:100], label = 'persistence prediction')
plt.plot(y_test[0:100], label = 'actual temp')
plt.legend()
plt.show()

print(f"model mse: {model_mse}")
print(f" persistence mse:{persistence_mse}")






