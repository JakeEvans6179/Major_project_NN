from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
PARQUET_FILE = Path("saved_householddata.parquet")
DATA_FOLDER = Path(r"Partitioned LCL Data")

COMMON_END = pd.Timestamp("2014-02-28 00:00:00")
WINDOW_DAYS = 800
COMMON_START = COMMON_END - pd.Timedelta(days=WINDOW_DAYS)

COVERAGE_THRESHOLD = 0.99
RANDOM_SEED = 69
N_HOUSES = 100

SELECTED_IDS_CSV = Path("selected_100_houses_fixed800d.csv")
GOOD_HOUSES_CSV = Path("good_houses_fixed800d.csv")
HOUSEHOLD_STATS_CSV = Path("household_stats_all_std.csv")
WINDOW_STATS_CSV = Path("window_stats_fixed800d.csv")
SELECTED_RAW_PARQUET = Path("selected_100_households_raw_fixed800d.parquet")

print("Common start:", COMMON_START)
print("Common end:  ", COMMON_END)

# ============================================================
# STEP 1: LOAD CLEANED DATA OR BUILD IT
# ============================================================
if PARQUET_FILE.exists():
    print(f"\nFound {PARQUET_FILE.name}. Loading data directly...")
    all_df = pd.read_parquet(PARQUET_FILE)

else:
    print(f"\n{PARQUET_FILE.name} not found. Processing raw CSVs...")

    csv_files = sorted(DATA_FOLDER.rglob("*.csv"))
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

    all_df = pd.concat(dfs, ignore_index=True)

    # remove exact duplicates
    all_df = all_df.drop_duplicates(subset=["LCLid", "stdorToU", "DateTime", "kwh"])

    # stdorToU no longer needed after filtering
    all_df = all_df.drop(columns=["stdorToU"])

    all_df.to_parquet(PARQUET_FILE, index=False)
    print(f"Finished processing and saved to {PARQUET_FILE.name}")

# defensive type conversion in case parquet was loaded
all_df["DateTime"] = pd.to_datetime(all_df["DateTime"], errors="coerce")
all_df["kwh"] = pd.to_numeric(all_df["kwh"], errors="coerce")

# remove rows with missing household id or timestamp
all_df = all_df.dropna(subset=["LCLid", "DateTime"]).copy()

print("\n--- Raw/Cleaned Data Summary ---")
print(all_df.head())
print("Shape:", all_df.shape)
print("Unique houses:", all_df["LCLid"].nunique())

# ============================================================
# STEP 2: SORT
# ============================================================
print("\nSorting by LCLid and DateTime...")
all_df = all_df.sort_values(["LCLid", "DateTime"]).reset_index(drop=True)

# ============================================================
# STEP 3: BUILD FULL-HOUSE STATS (OVER ENTIRE AVAILABLE HISTORY)
# ============================================================
print("\nCalculating household stats over full available history...")

household_stats = (
    all_df.groupby("LCLid")
    .agg(
        First_timestep=("DateTime", "min"),
        Last_timestep=("DateTime", "max"),
        Valid_count=("kwh", lambda x: x.notna().sum())
    )
    .reset_index()
    .rename(columns={"LCLid": "Household_id"})
)

household_stats["Total_count"] = (
    ((household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds() / (30 * 60))
    .round()
    .astype(int)
    + 1
)

household_stats["Coverage"] = household_stats["Valid_count"] / household_stats["Total_count"]

household_stats["Span_days"] = (
    (household_stats["Last_timestep"] - household_stats["First_timestep"]).dt.total_seconds()
    / (24 * 60 * 60)
)

print(household_stats.head())
print("Total houses in full stats:", len(household_stats))

# save full stats
household_stats.to_csv(HOUSEHOLD_STATS_CSV, index=False)

# ============================================================
# STEP 4: CHECK COMMON END DATES
# ============================================================
print("\nMost common full-history end dates:")
print(household_stats["Last_timestep"].value_counts().head(20))

# ============================================================
# STEP 5: REQUIRE HOUSES TO FULLY COVER THE FIXED COMMON WINDOW
# ============================================================
print("\nFiltering houses that fully cover the fixed common window...")

fixed_window_houses = household_stats[
    (household_stats["First_timestep"] <= COMMON_START) &
    (household_stats["Last_timestep"] >= COMMON_END)
].copy()

print("Houses fully covering window:", len(fixed_window_houses))

eligible_ids = fixed_window_houses["Household_id"].tolist()

# keep only rows inside the fixed window, and only from houses that fully cover it
window_df = all_df[
    (all_df["LCLid"].isin(eligible_ids)) &
    (all_df["DateTime"] >= COMMON_START) &
    (all_df["DateTime"] <= COMMON_END)
].copy()

print("Window dataframe shape:", window_df.shape)
print("Unique houses in fixed window:", window_df["LCLid"].nunique())

# ============================================================
# STEP 6: REMOVE HOUSES WITH OFF-GRID TIMESTAMPS IN THE WINDOW
# We expect timestamps exactly on :00 or :30 with second==0
# ============================================================
print("\nChecking for off-grid timestamps inside the fixed window...")

off_grid_mask = ~(
    window_df["DateTime"].dt.minute.isin([0, 30]) &
    (window_df["DateTime"].dt.second == 0)
)

bad_time_rows = window_df[off_grid_mask].copy()
bad_time_houses = bad_time_rows["LCLid"].unique().tolist()

print("Rows with off-grid timestamps:", len(bad_time_rows))
print("Houses with off-grid timestamps:", len(bad_time_houses))

if len(bad_time_houses) > 0:
    print("Example bad rows:")
    print(bad_time_rows.head(20))

window_df = window_df[~window_df["LCLid"].isin(bad_time_houses)].copy()

print("Window dataframe shape after removing off-grid houses:", window_df.shape)
print("Unique houses after removing off-grid houses:", window_df["LCLid"].nunique())

# ============================================================
# STEP 7: COMPUTE COVERAGE INSIDE THE SAME FIXED WINDOW
# ============================================================
print("\nComputing fixed-window coverage stats...")

expected_index = pd.date_range(start=COMMON_START, end=COMMON_END, freq="30min")
expected_count = len(expected_index)

print("Expected half-hour readings per house in fixed window:", expected_count)

window_stats = (
    window_df.groupby("LCLid")
    .agg(
        Valid_count=("kwh", lambda x: x.notna().sum()),
        Unique_timestamps=("DateTime", "nunique")
    )
    .reset_index()
    .rename(columns={"LCLid": "Household_id"})
)

window_stats["Total_count"] = expected_count
window_stats["Coverage"] = window_stats["Valid_count"] / window_stats["Total_count"]
window_stats["Timestamp_coverage"] = window_stats["Unique_timestamps"] / window_stats["Total_count"]

print(window_stats.head())
print(window_stats["Coverage"].describe())

# save fixed-window stats
window_stats.to_csv(WINDOW_STATS_CSV, index=False)

# ============================================================
# STEP 8: QUALITY FILTER
# ============================================================
print(f"\nApplying quality filter: Coverage > {COVERAGE_THRESHOLD}")

good_houses = window_stats[window_stats["Coverage"] > COVERAGE_THRESHOLD].copy()

print("Good houses after fixed-window filter:", len(good_houses))
print(good_houses.head())

good_houses.to_csv(GOOD_HOUSES_CSV, index=False)

if len(good_houses) < N_HOUSES:
    raise ValueError(
        f"Only {len(good_houses)} houses passed the filter, cannot sample {N_HOUSES}."
    )

# ============================================================
# STEP 9: RANDOMLY SAMPLE 100 HOUSES WITH FIXED SEED
# ============================================================
print(f"\nRandomly sampling {N_HOUSES} houses with seed {RANDOM_SEED}...")

rng = np.random.default_rng(RANDOM_SEED)

selected_100 = rng.choice(
    good_houses["Household_id"].to_numpy(),
    size=N_HOUSES,
    replace=False
)

selected_100 = pd.DataFrame({"Household_id": selected_100})

print(selected_100.head())
print("Selected houses:", len(selected_100))

selected_100.to_csv(SELECTED_IDS_CSV, index=False)

# ============================================================
# STEP 10: SAVE RAW DATA FOR JUST THE SELECTED 100 HOUSES
# ============================================================
selected_ids = selected_100["Household_id"].tolist()

selected_df = window_df[window_df["LCLid"].isin(selected_ids)].copy()

print("\nSelected raw dataframe summary:")
print(selected_df.head())
print("Shape:", selected_df.shape)
print("Unique selected houses:", selected_df["LCLid"].nunique())

selected_df.to_parquet(SELECTED_RAW_PARQUET, index=False)

print("\nSaved files:")
print(" -", PARQUET_FILE)
print(" -", HOUSEHOLD_STATS_CSV)
print(" -", WINDOW_STATS_CSV)
print(" -", GOOD_HOUSES_CSV)
print(" -", SELECTED_IDS_CSV)
print(" -", SELECTED_RAW_PARQUET)