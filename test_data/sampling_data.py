from pathlib import Path
import pandas as pd
import numpy as np

'''
Script for accessing household data and filtering
Filter out ToU and only use standard tarrif data

Remove duplicate readings, calculate total duration of each house and compute coverage ratio (ratio of valid readings/ total expected readings)

Enabling us to select list of 100 suitable households for running testing on
'''
# Define the path for your saved Parquet file
parquet_file = Path("saved_householddata.parquet")

# IF STATEMENT: Check if the file already exists
if parquet_file.exists():
    print(f"Found {parquet_file.name} Loading data directly")
    # Instantly load the compiled data
    all_df = pd.read_parquet(parquet_file)

else:
    print(f"{parquet_file.name} not found. Processing raw CSVs...")

    # folder containing all partitioned CSVs
    data_folder = Path(r"Partitioned LCL Data")

    # find all csv files recursively
    csv_files = sorted(data_folder.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dfs = []

    for i, f in enumerate(csv_files, start=1):
        print(f"Loading {i}/{len(csv_files)}: {f.name}")

        temp = pd.read_csv(
            f,
            usecols=["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "],
            low_memory=False
        )

        temp = temp.rename(columns={"KWH/hh (per half hour) ": "kwh"})

        # convert types
        temp["DateTime"] = pd.to_datetime(temp["DateTime"], errors="coerce")
        temp["kwh"] = pd.to_numeric(temp["kwh"], errors="coerce")

        # keep only standard tariff houses
        temp = temp[temp["stdorToU"] == "Std"].copy()

        dfs.append(temp)

    # combine into one dataframe
    all_df = pd.concat(dfs, ignore_index=True)

    # remove exact duplicates
    all_df = all_df.drop_duplicates(subset=["LCLid", "stdorToU", "DateTime", "kwh"])

    # drop std rating

    all_df = all_df.drop(columns=["stdorToU"])

    # Save to parquet file so the IF statement catches it next time
    all_df.to_parquet(parquet_file, index=False)
    print("Finished processing and saved to Parquet.")


# --- Regardless of how it was loaded, your data is now ready here ---
print("\n--- Data Summary ---")
print(all_df.head())
print("Shape:", all_df.shape)
print("Number of unique houses:", all_df["LCLid"].nunique())

print("Extracted house data")
print(all_df)

#convert kwh to numerical values
all_df['kwh'] = pd.to_numeric(all_df['kwh'], errors='coerce')


#testing = all_df[all_df['LCLid'] == 'MAC000022']
#print(testing)



print("Sorting household data by houseid and DateTime")
#sort by household id, then by dateTime
#Reset row number at each new household
all_df = all_df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

print("Sorted house data --> house id --> DataTime")
print(all_df)


print("Calculating household stats (Duration, coverage)")
household_stats = (
    all_df.groupby("LCLid") #takes dataset and splits into smaller chunks (one per house)
    .agg(       #extracts summary statistics for each house
        First_timestep=("DateTime", "min"), #gets first (min) datetime house was recorded
        Last_timestep=("DateTime", "max"),
        Valid_count=("kwh", lambda x: x.notna().sum())      #counts number of rows with actual numbers (ignore NaN/ missing values)
    ).reset_index().rename(columns={"LCLid": "Household_id"})   #rename column to Household_id
)


#print(household_stats)

household_stats["Total_count"] = (
    ((household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds() / (30 * 60)) #How many readings house should have in the interval
    .round()
    .astype(int)
    + 1 #Add one to account for initial starting point
)

household_stats["Coverage"] = household_stats["Valid_count"] / household_stats["Total_count"]   #find ratio of readings with data to total readings expected in period

# record length in days
household_stats["Span_days"] = (
    (household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds()
    / (24 * 60 * 60)
)

print("Household comparison stats:")
print(household_stats) #all house metrics are calculated

#Find households within the filters (>800 days duration, >0.99 coverage rating)
good_houses = household_stats[(household_stats['Coverage'] > 0.99) & (household_stats["Span_days"] > 800)]


#print(good_houses)

print("Randomly selecting household id from eligible list")
rng = np.random.default_rng(69)


# make sure you have at least 100
print("Eligible houses:", len(good_houses))

selected_100 = rng.choice(
    good_houses["Household_id"].to_numpy(),
    size=100,
    replace=False
)

selected_100 = pd.DataFrame({"Household_id": selected_100})

print(selected_100.head())
print("Selected houses:", len(selected_100))


