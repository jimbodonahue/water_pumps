
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd # geospatial data processing
import matplotlib.pyplot as plt # standard plotting packages
import seaborn as sns
import os
import seaborn as sns
# Build path to data folder
path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Read the files
train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
labels = pd.read_csv(os.path.join(path, 'train_labels.csv'))

#Initial exploring,
print("Train shape:", train.shape)
print("Preview of train data:")
print(train.head(10))
print("\nTrain columns:")
print(train.columns)

print("Test shape:", test.shape)
print("Labels shape:", labels.shape)
print("Preview of labels:")
print(labels.head())

print(train.dtypes.value_counts())
print(train.describe().T)

# Replace 0s with NaN where 0 means "missing" or invalid
df = train.merge(labels, on='id') # Merge tain and lable
df['construction_year'] = df['construction_year'].replace(0, np.nan)
df['gps_height'] = df['gps_height'].replace([0, -90], np.nan)
df['longitude'] = df['longitude'].replace(0, np.nan)
print("Missing values after cleaning invalid numeric entries:")
print(df[['construction_year', 'gps_height', 'longitude']].isnull().sum())


