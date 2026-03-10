import pandas as pd


#df = pd.read_csv('CC_LCL-FullData.csv')
df=pd.read_csv('Partitioned LCL Data/Small LCL Data/LCL-June2015v2_6.csv')



selected_house = df[df['LCLid'] == 'MAC000206'].copy() #only take houses with MAC000002 ID into new dataframe

print(selected_house.head())

selected_house = selected_house.rename(columns={'KWH/hh (per half hour) ': 'kwh_hh'})

#remove any duplicate rows
selected_house = selected_house.drop_duplicates(
    subset=['LCLid', 'stdorToU', 'DateTime', 'kwh_hh'],
    keep='first'
)


selected_house['DateTime'] = pd.to_datetime(selected_house['DateTime']) #set it to date time object
selected_house = selected_house.set_index('DateTime')   #set index as datatime

print(selected_house.head())
print(selected_house.shape)

print(selected_house[:24])
#selected_house = selected_house.iloc[:-1]


#convert data to numbers (originally strings)
selected_house['kwh_hh'] = pd.to_numeric(selected_house['kwh_hh'], errors='coerce')

selected_house.index = selected_house.index - pd.Timedelta(minutes=30)  #shift data forward by 30mins to allow summation to sum correctly
hourly = pd.DataFrame(selected_house['kwh_hh'].resample('h').sum()) #sum hourly


print(hourly)
print(hourly.shape)





