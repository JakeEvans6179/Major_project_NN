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


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)




#temp_mean = np.mean(x_train[:,:,0]) #take all training examples, all lists, and take the first index (temperature value)
#temp_std = np.std(x_train[:,:,0])



model2 = Sequential([
    tf.keras.Input(shape=(24,5)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'linear')
])

model2.summary()

model2.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])

cp = ModelCheckpoint('model2_checkpoint.keras', save_best_only=True)

#The Early Stopper: Stop if we don't improve for 5 hours (epochs)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model2.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=[cp, es] #only saves model (checkpoint) if it performs the best so far, and stops training if MSE doesn't decrease after 5 rounds
)



model2 = load_model('model2_checkpoint.keras')

#run inference using test set and plot
plot_predictions(model2, x_test, y_test)



