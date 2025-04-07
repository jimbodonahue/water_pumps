# water_pumps
## Water Pump Functionality Prediction

# Team: 
### Fatemeh Ebrahimi
### Kateryna Ponomarova
### James Donahue

## Project Overview
This project aims to develop machine learning models to predict the functionality status of water pumps in Tanzania based on factors like location, water quality, management structure, and technical specifications. It addresses the challenge of identifying functional, repair-needed, and non-functional pumps to enhance maintenance and ensure access to clean water for communities.

## Competition Link
[DrivenData - Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)


## Project Goals
1. Develop classification models to predict three target classes:
   - Functional water pumps
   - Water pumps that need repair
   - Non-functional water pumps
2. Analyze which factors most strongly influence water pump functionality
3. Create geospatial visualizations to communicate findings effectively
4. Deploy a simple dashboard to showcase models and insights

## Data Description

### Dataset Source
The data comes from Taarifa and the Tanzanian Ministry of Water, which aggregates information about water points across Tanzania.
# Features in this dataset

- `amount_tsh` - Total static head (amount water available to waterpoint)
- `date_recorded` - The date the row was entered
- `funder` - Who funded the well
- `gps_height` - Altitude of the well
- `installer` - Organization that installed the well
- `longitude` - GPS coordinate
- `latitude` - GPS coordinate
- `wpt_name` - Name of the waterpoint if there is one
- `num_private` -
- `basin` - Geographic water basin
- `subvillage` - Geographic location
- `region` - Geographic location
- `region_code` - Geographic location (coded)
- `district_code` - Geographic location (coded)
- `lga` - Geographic location
- `ward` - Geographic location
- `population` - Population around the well
- `public_meeting` - True/False
- `recorded_by` - Group entering this row of data
- `scheme_management` - Who operates the waterpoint
- `scheme_name` - Who operates the waterpoint
- `permit` - If the waterpoint is permitted
- `construction_year` - Year the waterpoint was constructed
- `extraction_type` - The kind of extraction the waterpoint uses
- `extraction_type_group` - The kind of extraction the waterpoint uses
- `extraction_type_class` - The kind of extraction the waterpoint uses
- `management` - How the waterpoint is managed
- `management_group` - How the waterpoint is managed
- `payment` - What the water costs
- `payment_type` - What the water costs
- `water_quality` - The quality of the water
- `quality_group` - The quality of the water
- `quantity` - The quantity of water
- `quantity_group` - The quantity of water
- `source` - The source of the water
- `source_type` - The source of the water
- `source_class` - The source of the water
- `waterpoint_type` - The kind of waterpoint
- `waterpoint_type_group` - The kind of waterpoint

# 📊 Initial Data Exploration & First Impressions  

## ✅ Dataset Overview
- The training dataset contains **59,400 rows** and **40 features**.
- A separate labels file includes the `status_group` target column (pump functionality).

## 🧾 Data Types Summary

| Type     | Count | Description                             |
|----------|-------|-----------------------------------------|
| `object` | 30    | Categorical or string-type columns      |
| `int64`  | 7     | Integer columns (e.g., year, region)    |
| `float64`| 3     | Float columns (e.g., coordinates)       |


## 🧠 Key Observations from `.describe()` and `.head()`

- **Skewed distributions** observed in  `amount_tsh`, `population`, and `construction_year` features.
- **Zero values** in `gps_height`, `longitude`, and `latitude` likely indicate **missing or invalid data**.
- `num_private` appears to have **mostly zero values**, and may be **dropped** if uninformative.
-  Most features are **categorical (`object`)** – 30 out of 40 columns
-   Repetitive features like `extraction_type`, `extraction_type_group`, and `extraction_type_class` might offer **redundant information**
  
### 📌 Summary
- The dataset has **less unique information than it first appears**
- Careful handling of missing values, feature selection, and encoding will be needed before modeling
- Geospatial features (lat/lon) offer opportunities for external enrichment like weather or population data

## 🗺️ Geospatial Visualization

We created a basic geospatial map of water pumps across Tanzania using `geopandas`.

### Key Steps:
- Used the shapefile from https://www.nbs.go.tz/statistics/topic/gis
- Cleaned the pump data by removing rows with missing latitude or longitude
- Converted the cleaned data into a GeoDataFrame
- Plotted pump locations over a population-colored map of Tanzania
