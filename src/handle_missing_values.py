# handle_missing_values.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load cleaned base data from data_cleaning.py
path = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(path,'cleaned_data_V1.csv'))

# Analyze missingness
missing = df.isnull().sum()
print("Missing values_construction_year:\n", missing[missing > 0])
# Handelling missing year
# Save original missing count
missing_before = df['construction_year'].isna().sum()
print("Missing before:", missing_before)
# Pumps in the same region likely have similar construction years
# Same funder or installer may follow same timeline
# Fill missing years with the median by region + installer
# If construction_year is missing, we could use date_recorded.year as an upper bound for imputation.
df['date_recorded'] = pd.to_datetime(df['date_recorded'])
df['recorded_year'] = df['date_recorded'].dt.year

#visualize the relationship between construction_year and date_recorded to see how they are related
valid_years = df[df['construction_year'].notna()].copy()
valid_years['difference_year'] = valid_years['recorded_year']-valid_years['construction_year']

plt.figure(figsize=(8, 5))
sns.histplot(valid_years['difference_year'], bins=30, kde=True)
plt.axvline(5, color='red', linestyle='--', label='5-Year Gap')
plt.show()
#Impute using region + installer
df['construction_year'] = df.groupby(['region', 'installer'])['construction_year'].transform(
    lambda x: x.fillna(x.median())
)
#Impute using region only (for rows still missing)
df['construction_year'] = df.groupby('region')['construction_year'].transform(
    lambda x: x.fillna(x.median())
)
#Use recorded year - 5
df['construction_year'] = df['construction_year'].fillna(df['recorded_year'] - 5)
missing_after = df['construction_year'].isna().sum()
print("Missing after_construction_year:", missing_after)
filled_count = missing_before - missing_after
print("Filled_construction_year:", filled_count)
df.to_csv("data/cleaned_data_filled.csv", index=False)

print(df['longitude'].describe())
print("latitudeb and longitude missing value before:",df[['latitude', 'longitude']].isnull().sum())
#filling the missing longitude values using info from other pumps in the same region and district
df['longitude']= df.groupby(['region', 'district_code'])['longitude'].transform(
    lambda x: x.fillna(x.median())
)

#Filling the remaining missing values with the median longitude of the entire region:
df['longitude'] = df.groupby('region')['longitude'].transform(
    lambda x: x.fillna(x.median())
)

print("Missing longitude after filling:", df['longitude'].isna().sum())
# Save updated version
df.to_csv("data/cleaned_data_filled_V2.csv", index=False)
print("Cleaned data saved to data/cleaned_data_filled_V2.csv")
