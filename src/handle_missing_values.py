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
missing_before = df['construction_year'].isna().sum()
print("Missing before:", missing_before)

# Date info
df['date_recorded'] = pd.to_datetime(df['date_recorded'])
df['recorded_year'] = df['date_recorded'].dt.year

# Visualize construction year gap
valid_years = df[df['construction_year'].notna()].copy()
valid_years['difference_year'] = valid_years['recorded_year'] - valid_years['construction_year']

plt.figure(figsize=(8, 5))
sns.histplot(valid_years['difference_year'], bins=30, kde=True)
plt.axvline(5, color='red', linestyle='--', label='5-Year Gap')
plt.show()

# Impute construction year
df['construction_year'] = df.groupby(['region', 'installer'])['construction_year'].transform(lambda x: x.fillna(x.median()))
df['construction_year'] = df.groupby('region')['construction_year'].transform(lambda x: x.fillna(x.median()))
df['construction_year'] = df['construction_year'].fillna(df['recorded_year'] - 5)

missing_after = df['construction_year'].isna().sum()
print("Missing after_construction_year:", missing_after)
filled_count = missing_before - missing_after
print("Filled_construction_year:", filled_count)
df.to_csv("data/cleaned_data_filled.csv", index=False)

# Handle GPS coordinates
print(df['longitude'].describe())
print("latitudeb and longitude missing value before:", df[['latitude', 'longitude']].isnull().sum())

# Convert to numeric in case of weird types
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop invalid latitudes: outside Tanzania range or near zero
df['latitude'] = df['latitude'].apply(lambda x: np.nan if pd.isna(x) or abs(x) < 0.1 or x < -15 or x > 0 else x)
df['longitude'] = df['longitude'].apply(lambda x: np.nan if pd.isna(x) or abs(x) < 10 or x < 29 or x > 42 else x)

# Drop rows where lat/lon still missing
df = df.dropna(subset=['latitude', 'longitude'])
print("Remaining rows after dropping invalid GPS data:", len(df))

# Save updated version
df.to_csv("data/cleaned_data_filled_V2.csv", index=False)
print("Cleaned data saved to data/cleaned_data_filled_V2.csv")

# filling missing values for gps_height
df['gps_height'] = df['gps_height'].apply(lambda x: np.nan if x <= 0 else x)
missing_gps_before = df['gps_height'].isna().sum()
print("Missing before gps_height:", missing_gps_before)

df['gps_height'] = df.groupby('basin')['gps_height'].transform(lambda x: x.fillna(x.median()))
df['gps_height'] = df.groupby('region')['gps_height'].transform(lambda x: x.fillna(x.median()))
missing_gps_after = df['gps_height'].isna().sum()
print("Missing after gps_height:", missing_gps_after)
print("Filled gps_height:", missing_gps_before - missing_gps_after)
df.to_csv("data/cleaned_data_filled_V3.csv", index=False)
print("Saved updated dataset to data/cleaned_data_filled_V3.csv")

# filling population
df['population'] = df['population'].replace(0, np.nan)
missing_pop_before = df['population'].isna().sum()
print("Missing population before:", missing_pop_before)

df['population'] = df.groupby('district_code')['population'].transform(lambda x: x.fillna(x.median()))
df['population'] = df.groupby('region')['population'].transform(lambda x: x.fillna(x.median()))

df.to_csv("data/cleaned_data_filled_V4.csv", index=False)

sns.histplot(df['population'], bins=50, kde=True)
plt.xlim(0, 10000)
plt.title("Population Distribution Around Water Pumps")
plt.savefig("outputs/population_distribution_before_clipping.png", dpi=300, bbox_inches='tight')
plt.show()

df['population'] = df['population'].clip(upper=1999)
sns.histplot(df['population'], bins=50, kde=True)
plt.xlim(0, 10000)
plt.title("Population Distribution Around Water Pumps afterclipping")
plt.savefig("outputs/population_distribution_after_clipping.png", dpi=300, bbox_inches='tight')
df.to_csv("data/cleaned_data_filled_V5.csv", index=False)
plt.show()
