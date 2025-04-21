# Imports and Setup
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

data_path = os.path.join(os.getcwd(), 'data')
out_path = os.path.join(os.getcwd(),'outputs')     # For the output


# Read the files
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))


# Merge training labels
df = pd.merge(train, labels, on='id')
print('Files successfully loaded and merged!')
# Replace 0s with NaN where 0 means "missing" or invalid

df['construction_year'] = df['construction_year'].replace(0, np.nan)
df['gps_height'] = df['gps_height'].replace([0, -90], np.nan)
df['longitude'] = df['longitude'].replace(0, np.nan)
print("Missing values after cleaning invalid numeric entries:")
print(df[['construction_year', 'gps_height', 'longitude']].isnull().sum())
missing_before = df['construction_year'].isna().sum()
# Save original missing count

print("Missing construction_year before:", missing_before)
# Pumps in the same region likely have similar construction years
# Same funder or installer may follow same timeline
# Fill missing years with the median by region + installer
# If construction_year is missing, we could use date_recorded.year as an upper bound for imputation.
df['date_recorded'] = pd.to_datetime(df['date_recorded'], errors='coerce')
df['recorded_year'] = df['date_recorded'].dt.year
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
df.to_csv(os.path.join(out_path, "cleaned_data_filled.csv"), index=False)
print(df['longitude'].describe())
print("Missing values in latitude and longitude:",df[['latitude', 'longitude']].isnull().sum())
