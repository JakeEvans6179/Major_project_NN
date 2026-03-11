from pathlib import Path
import pandas as pd

# ----------------------------
# SETTINGS
# ----------------------------
HOUSE_ID = "MAC000002"
DATA_FOLDER = Path(r"Partitioned LCL Data")   # change if needed

# ----------------------------
# LOAD ONLY THIS HOUSE FROM ALL CSVs
# ----------------------------
csv_files = sorted(DATA_FOLDER.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []

for i, f in enumerate(csv_files, start=1):
    temp = pd.read_csv(
        f,
        usecols=["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "],
        low_memory=False
    )

    # keep only this house
    temp = temp[temp["LCLid"] == HOUSE_ID].copy()

    if not temp.empty:
        temp = temp.rename(columns={"KWH/hh (per half hour) ": "kwh"})
        dfs.append(temp)
        print(f"Loaded {f.name}: {len(temp)} rows for {HOUSE_ID}")

if not dfs:
    raise ValueError(f"No rows found for {HOUSE_ID}")

raw_house = pd.concat(dfs, ignore_index=True)

print("\n--- RAW HOUSE DATA ---")
print(raw_house.head())
print("Raw total rows:", len(raw_house))

# ----------------------------
# APPLY SAME CLEANING LOGIC AS YOUR PIPELINE
# ----------------------------

# keep only Std tariff if you want to match your main pipeline
raw_house = raw_house[raw_house["stdorToU"] == "Std"].copy()

print("\nRows after Std tariff filter:", len(raw_house))

# convert types the same way as your main script
raw_house["DateTime"] = pd.to_datetime(raw_house["DateTime"], errors="coerce")
raw_house["kwh"] = pd.to_numeric(raw_house["kwh"], errors="coerce")

# count missing / invalid kwh
invalid_kwh_count = raw_house["kwh"].isna().sum()
print("Invalid / missing kwh rows:", invalid_kwh_count)

# count duplicate rows using the same columns as your main script
dup_mask = raw_house.duplicated(
    subset=["LCLid", "stdorToU", "DateTime", "kwh"],
    keep="first"
)

duplicate_count = dup_mask.sum()
print("Duplicate rows to be removed:", duplicate_count)

# cleaned house data
clean_house = raw_house.drop_duplicates(
    subset=["LCLid", "stdorToU", "DateTime", "kwh"],
    keep="first"
).copy()

print("Rows after duplicate removal:", len(clean_house))

# valid count after cleaning
valid_count_after_cleaning = clean_house["kwh"].notna().sum()
print("Valid_count after cleaning:", valid_count_after_cleaning)

# ----------------------------
# CHECK THE ACCOUNTING
# ----------------------------
print("\n--- CHECK ---")
print("Raw rows                     :", len(raw_house))
print("Minus duplicates removed     :", duplicate_count)
print("Rows after duplicate removal :", len(clean_house))
print("Minus invalid/missing kwh    :", clean_house['kwh'].isna().sum())
print("Final valid_count            :", valid_count_after_cleaning)

# ----------------------------
# OPTIONAL: SHOW THE DUPLICATE ROWS
# ----------------------------
duplicate_rows = raw_house[dup_mask].copy()

print("\n--- SAMPLE DUPLICATE ROWS ---")
print(duplicate_rows.head(20))

# save them if you want to inspect in Excel
duplicate_rows.to_csv(f"{HOUSE_ID}_duplicate_rows.csv", index=False)
clean_house.to_csv(f"{HOUSE_ID}_cleaned_rows.csv", index=False)

print(f"\nSaved duplicate rows to: {HOUSE_ID}_duplicate_rows.csv")
print(f"Saved cleaned rows to:   {HOUSE_ID}_cleaned_rows.csv")