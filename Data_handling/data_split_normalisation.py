import numpy as np
import pandas as pd
from pathlib import Path


'''
Data has been preprocessed in test_pipeline

Now split into training, validation and test data
Apply min max scaling from training set on all sets

Run training and evaluate average performance across all household data
'''

hourly_path = Path("selected_100_households_hourly_processed.parquet")

df = pd.read_parquet(hourly_path)



train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert round(train_ratio + val_ratio + test_ratio) == 1.0


df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce") #set to DateTime object

#sort by houseid then by datetime within each houseid
df = df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

pd.set_option('display.max_columns', None)
print(df)
print("Unique houses:", df["LCLid"].nunique())
print("Shape:", df.shape)


#split data into train val test sets
house_splits = {}   #used to store each houses training, validation and test set
all_train_kwh = []  #used to collect all kwh data from training sets to find global max and global min

house_ids = sorted(df["LCLid"].unique())        #sort house ids in ascending order

for house_id in house_ids:
    house = df[df["LCLid"] == house_id].copy().sort_values("DateTime")

    n = len(house)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = house.iloc[:train_end].copy()
    val_df = house.iloc[train_end:val_end].copy()
    test_df = house.iloc[val_end:].copy()

    house_splits[house_id] = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    all_train_kwh.append(train_df["kwh"])


#combine all training kwh values across all houses --> used to calculate global min max values for scaling (on training set)
all_train_kwh = pd.concat(all_train_kwh, ignore_index=True)

print(all_train_kwh)

global_kwh_min = all_train_kwh.min()
global_kwh_max = all_train_kwh.max()

print("\nGlobal training kwh min:", global_kwh_min)
print("Global training kwh max:", global_kwh_max)

if global_kwh_max == global_kwh_min:
    raise ValueError("Global kwh min and max are equal; cannot apply min-max scaling.")

# --------------------------------------------------
# MIN-MAX SCALE KWH ONLY
# --------------------------------------------------
def minmax_scale_kwh(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)


#apply min max scaling for all sets (train, val, test)
for house_id in house_ids:
    house_splits[house_id]["train"]["kwh"] = minmax_scale_kwh(
        house_splits[house_id]["train"]["kwh"], global_kwh_min, global_kwh_max
    )
    house_splits[house_id]["val"]["kwh"] = minmax_scale_kwh(
        house_splits[house_id]["val"]["kwh"], global_kwh_min, global_kwh_max
    )
    house_splits[house_id]["test"]["kwh"] = minmax_scale_kwh(
        house_splits[house_id]["test"]["kwh"], global_kwh_min, global_kwh_max
    )



train_list = []
val_list = []
test_list = []

#Loop through every house and append its chunks to the correct list
#used later for centralised training
# --- For Later ---
for h in house_ids:
    train_list.append(house_splits[h]["train"])
    val_list.append(house_splits[h]["val"])
    test_list.append(house_splits[h]["test"])

#print(train_list)
#connect the populated lists together into one dataframe
train_all = pd.concat(train_list, ignore_index=True)
val_all = pd.concat(val_list, ignore_index=True)
test_all = pd.concat(test_list, ignore_index=True)

print("\nTrain shape:", train_all.shape)
print("Val shape:", val_all.shape)
print("Test shape:", test_all.shape)


print(train_all)

#sanity check, min = 0, max = 1
print("\nScaled train kwh range:")
print(train_all["kwh"].min(), train_all["kwh"].max())

# ------

print("\nExample one house split sizes:")
example_house = house_ids[0]
print(example_house,
      len(house_splits[example_house]["train"]),
      len(house_splits[example_house]["val"]),
      len(house_splits[example_house]["test"]))



#create one df with all train, split, val data for all households to save and run in training model
#loop through each house, loop through training, val and test data, append to list
#once all houses appended, concatenate into dataframe and save to parquet file
rows = []

for house_id in house_ids:
    train_df = house_splits[house_id]["train"].copy() #get all training examples for house id
    train_df["split"] = "train"         #add label train to column called split
    rows.append(train_df)               #append to list

    val_df = house_splits[house_id]["val"].copy()
    val_df["split"] = "val"
    rows.append(val_df)

    test_df = house_splits[house_id]["test"].copy()
    test_df["split"] = "test"
    rows.append(test_df)

splits_df = pd.concat(rows, ignore_index=True)
pd.set_option('display.max_columns', None)
print(splits_df)
print(splits_df.shape)
print(splits_df["split"].value_counts())

splits_df.to_parquet("selected_100_households_hourly_scaled_with_splits.parquet", index=False)

pd.DataFrame({
    "global_kwh_min": [global_kwh_min],
    "global_kwh_max": [global_kwh_max]
}).to_csv("global_kwh_scaler.csv", index=False)





