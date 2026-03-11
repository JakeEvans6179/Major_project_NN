import numpy as np
import pandas as pd

'''
Script for using the 100 households we determined from samping_data.py to train and compare different LSTM model architectures 
'''
rng = np.random.default_rng(69)

good_houses = pd.DataFrame({"Household_id": [1,2,3,4,5]})

# make sure you have at least 100
print("Eligible houses:", len(good_houses))

selected_100 = rng.choice(
    good_houses["Household_id"].to_numpy(),
    size=2,
    replace=False
)

selected_100 = pd.DataFrame({"Household_id": selected_100})

print(selected_100.head())
print("Selected houses:", len(selected_100))