import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.models import load_model


current_dir = os.getcwd() # get current directory
print(f"Current directory: {current_dir}")

extracted_folder_path = tf.keras.utils.get_file(
    fname = 'jena_climate_2009_2016.csv.zip', # Just the filename now!
    origin = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    cache_dir = current_dir,                  # Tell it to use your current folder
    cache_subdir = '.',                       # Tell it NOT to create a 'datasets' subfolder
    extract = True,                         #unzip file
)

# Join the folder path with the actual name of the CSV file inside it
csv_path = os.path.join(extracted_folder_path, "jena_climate_2009_2016.csv")
print(f"Dataset successfully saved to: {csv_path}")


df = pd.read_csv(csv_path)
# Tell Pandas to show all columns
pd.set_option('display.max_columns', None)

print(df.head()) # .head() ensures you still only print the top 5 rows

df = df[5::6] #start at 5th row, take every 6th row (ie every hour) [start : stop: step] --> no stop = go to end of dataset
#print(df.head())
print(df)

df.index = pd.to_datetime(df['Date Time'], format = '%d.%m.%Y %H:%M:%S') #converts datetime text on dataset to Datetime object, and sets our label index to these objects
print(df[:26])  #can still call the indexes using row number even though date time is the official label


#just doing univariate LSTM so just get temp
temp = df['T (degC)']
print(temp.head())  #now is a pandas series

temp.plot()

plt.show()
'''
#need to organise data into input and outputs 
In this example, we take 5 values before the hour we want to predict
e.g. 1,2,3,4,5 --> predicts 6
    2,3,4,5,6 --> predicts 7
    3,4,5,6,7 --> predicts 8
    
So to rearrange data to be:
    [[1, 2, 3, 4, 5]] [6]   where   [[1,2,3,4,5]]
    [[2, 3, 4, 5, 6]] [7]           [[2,3,4,5,6]] is our feature X matrix 
    
    [6]
    [7] is our output Y vector
etc  
    ---
    [[[1], [2], [3], [4], [5]]]
    [[[2], [3], [4], [5], [6]]] --> each variable should be wrapped in a list, in case we were doing multivariate forecasting 

'''
#create function to organise variables
def df_to_x_y(df, window_size = 5):
    df_as_np = df.to_numpy()    #convert to numpy array
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):

        row = [[a] for a in df_as_np[i:i+5]]    #takes the first 5 values e.g. 0,1,2,3,4 --> wraps series in a list and each element is wrapped by a list
        X.append(row)

        label = df_as_np[i+5]                   #takes the value one index after the row taken e.g. 5
        y.append(label)

    return np.array(X), np.array(y)

X,y = df_to_x_y(temp)   #convert

print(X.shape)
print(y.shape)

'''
Prepare for training --> split to train, val, test sets
'''

X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


'''
Create model
'''

model = Sequential([

    Input(shape=(5,1)),
    LSTM(64),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'linear')

])

model.summary()


#The Checkpoint: Save the 'Peak' version to a folder called 'model/'
cp = ModelCheckpoint('model_checkpoint.keras', save_best_only=True)

#The Early Stopper: Stop if we don't improve for 5 hours (epochs)
es = EarlyStopping(monitor='val_loss', patience=5)


model.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0001), metrics = [RootMeanSquaredError()])

# 3. Use BOTH in the callbacks list
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[cp, es] #only saves model (checkpoint) if it performs the best so far, and stops training if MSE doesn't decrease after 5 rounds
)


model = load_model('model_checkpoint.keras')

train_predictions = model.predict(X_test).flatten()

train_results = pd.DataFrame(data = {'prediction': train_predictions, 'target': y_test})
print(train_results)


plt.plot(train_results['prediction'][:10], label='Preds')
plt.plot(train_results['target'][:10], label='Actuals')
plt.legend()
plt.show()