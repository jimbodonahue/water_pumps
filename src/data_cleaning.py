
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

# Load the Tanzania boundary map
tanzania = gpd.read_file("gis/Tanzania.shp")
# Quick preview
print(tanzania.head())

# Filter out missing lat/lon rows
df_valid = df.dropna(subset=['longitude', 'latitude'])
# Create a GeoDataFrame for pumps
gdf_pumps = gpd.GeoDataFrame(
    df_valid,
    geometry=gpd.points_from_xy(df_valid.longitude, df_valid.latitude),
    crs="EPSG:4326"
)
gdf_pumps = gdf_pumps.to_crs(tanzania.crs)
fig, ax = plt.subplots(figsize=(4, 4))
# Plot Tanzania base

# Plot Tanzania base
tanzania.plot(ax= ax, column='POPULATION',legend=True)

# Plot pump points
gdf_pumps.plot(ax= ax, markersize=0.5, alpha=0.1, color='black')

plt.title("Water Pump Locations in Tanzania")
plt.axis('off')
plt.show()
fig.savefig("outputs/water_pumps_population_map.png", dpi=300, bbox_inches='tight')
