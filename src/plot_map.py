 
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
df = pd.read_csv(os.path.join(path, 'cleaned_data_filled_V2.csv'))

# Load the Tanzania boundary map
tanzania = gpd.read_file("gis/Tanzania.shp")

# Filter out missing lat/lon rows
df_valid = df.dropna(subset=['longitude', 'latitude'])
# Create a GeoDataFrame for pumps
gdf_pumps = gpd.GeoDataFrame(
    df_valid,
    geometry=gpd.points_from_xy(df_valid.longitude, df_valid.latitude),
    crs="EPSG:4326"
)
gdf_pumps = gdf_pumps.to_crs(tanzania.crs)
fig, ax = plt.subplots(figsize=(40, 40))

# Plot Tanzania base
tanzania.plot(ax= ax, column='POPULATION',legend=True, legend_kwds={"label": "Population in 2012", "orientation": "vertical"})

# Plot pump points
# Custom status colors
status_colors = {
    'functional': 'black',                 
    'functional needs repair': 'yellow',
    'non functional': 'orange'
}

for status, color in status_colors.items():
    gdf_pumps[gdf_pumps['status_group']== status].plot(ax= ax,column='status_group', markersize=1.5, alpha=0.7,color=color, label=status )

plt.title("Water Pump Locations in Tanzania(Imputed Coordinates)")
plt.legend(title="Pump Status")
plt.axis('off')
plt.show()
fig.savefig("outputs/water_pumps_map_V2.png", dpi=300, bbox_inches='tight')
