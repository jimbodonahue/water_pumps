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
- 
## 🧠 Observations from visualizing variables
- Some regions seem to have much better luck than others
- `quantity` could be a useful binary (enough, else), or even three?
- `management_group` does not seem to have an effect on the distribution. Nor does `permit`
- `quality_group` does not seem to bring much, as there are probably not enough "bad" observations. Similar with `waterpoint_type_group` (maybe?), `extraction_type_class`
- There's a strong case for a binary `payment` variable. Also `source`
- The watershed variabe `basin` might be good but may simply reflect seasonality captured in `quantity`
- 
## 🗺️ Geospatial Visualization

We created a basic geospatial map of water pumps across Tanzania using `geopandas`.

### Key Steps:
- Used the shapefile from https://www.nbs.go.tz/statistics/topic/gis
- Cleaned the pump data by removing rows with missing latitude or longitude
- Converted the cleaned data into a GeoDataFrame
- Plotted pump locations over a population-colored map of Tanzania
# Data cleaning and preparation

### 🧼 Missing Value Handling: Construction Year

- Replaced invalid `0` values in `construction_year` with `NaN`
- Imputed missing values in three steps:
  1. Used **median construction year** per **region + installer** combination
  2. For remaining missing values, used **region-level median**
  3. For any still missing, used the fallback: `recorded_year - 5`, based on data distribution
- This strategy filled all 20,709 missing entries using a context-aware approach
## 🧼 Missing Value Coordinate: `longitude` and `Longitude`
During data preprocessing, we identified a large number of water pumps with invalid or highly suspicious latitude values, such as -2e-08, which caused them to appear far outside of Tanzania’s geographic boundaries. Initially, we attempted to impute these missing GPS coordinates using grouped medians based on geographic features such as region, district_code, ward, and lga. However, this approach resulted in many identical coordinate clusters that still fell outside the Tanzanian borders.

To ensure the spatial integrity of the dataset, we ultimately decided to remove all rows with invalid or missing latitude/longitude values. Specifically, latitude values outside the valid range for Tanzania (between -15 and 0) or extremely close to zero (i.e., abs(latitude) < 0.1) were treated as invalid and excluded. This ensures that all remaining records represent real, mappable locations within Tanzania.
- Output saved as: `data/cleaned_data_filled_V2.csv` 
### 🧼 Missing Value Handling: `gps_height`
- Replaced invalid `gps_height` values (≤ 0) with `NaN`
- Total filled entries: **21,934** (~37% of dataset)
- Imputed values using a two-step geographic approach:
  1. Median per `basin` (hydrological unit)
  2. Median per `region` (fallback)
-  No values left missing
- Output saved as: 'data/cleaned_data_filled_V3.csv'
- This method ensures geographic consistency by reflecting local altitude patterns

### 🧼 Missing Value Handling: `population`
- Replaced invalid values (`0`) with `NaN`
- Total filled entries: **21,381** (~36% of the dataset)
- Used a two-step geographic imputation:
  1. Median per `district_code`
  2. Median per `region` (fallback)
- After imputation, all missing values were filled
- As an additional step, we capped population values at **2,500**, based on domain knowledge and national-level population density in Tanzania
  - This avoids skew from unusually large or erroneous values
  #### 📉 Population Clipping
- Capped population values at **1,999** based on:
  - Distribution analysis
  - Realistic rural/urban estimates for Tanzania
- This prevents model distortion caused by a small number of extreme outliers
- Saved before/after distribution plots in `outputs/` folder:
  - `population_distribution_before_clipping.png`
  - `population_distribution_after_clipping.png`
#### 📉 amount_tsh Capping

- Identified extreme outliers in the `amount_tsh` feature (some values > 100,000).
- To reduce the impact of these extreme values, **capped `amount_tsh` at 15,000**.
- Created a new feature called **`amount_tsh_capped`** containing the capped values.
- Saved the updated dataset as **`cleaned_data_filled_V5.csv`**.
- This step ensures that later modeling processes (especially regression models) are not dominated by rare extreme measurements.
### 🧼 Handling Missing Categorical Values

To ensure the dataset is clean and suitable for machine learning models, missing values in categorical columns were filled with a placeholder value `'Unknown'`. This allows us to retain all rows without dropping data or introducing statistical bias.


## 🔍 Exploratory Regression Analysis

As part of our initial data exploration, we ran a basic linear regression using `statsmodels` to examine how a few numeric features relate to the functionality of water pumps.

### ✅ What We Did

- Loaded cleaned data from `cleaned_data_filled_V6.csv`.

- Used five numerical features:
  - `amount_tsh_capped` (capped water availability)
  - `gps_height` (altitude)
  - `population` (population around the pump)
  - `construction_year` (normalized)
  - `num_private`
- Normalized (standardized) the following features:
  - `amount_tsh_capped`
  - `construction_year`
- Mapped the target variable `status_group` into numeric codes:
  - `2` = functional
  - `1` = functional needs repair
  - `0` = non functional
- Ran an OLS (Ordinary Least Squares) regression model using `statsmodels` with normalized features.

### 📊 Key Findings

- `amount_tsh_capped`, `gps_height`, `population`, and `construction_year` were statistically significant predictors (p < 0.05).
- `num_private` was **not statistically significant**.
- The most important predictors were `gps_height` and `amount_tsh_capped`.
- The model had a low R² (~0.024), meaning the features only explain about 2.4% of the variance in pump status.
- This low R² suggests that linear regression is not sufficient for this classification problem.

### ⚠️ Notes

- This analysis was purely exploratory — not meant for prediction.
- In future steps, a classification model (e.g., Random Forest, XGBoost) will be used to better capture nonlinear relationships and improve performance.

# Sprint 2

## 🛠️ Feature Engineering (Task 2.1)

As part of Ticket 2.1.1 and 2.1.2, we applied domain knowledge and data exploration insights to improve feature quality before modeling.

### ✅ Feature Creation (Ticket 2.1.1)
- Created `water_risk_score` by combining `amount_tsh_capped` (water quantity) and a numerical encoding of `water_quality`.
- Generated `pump_age` from `construction_year` to represent the pump's operational age.

### ✅ Feature Transformation (Ticket 2.1.2)
- Identified and handled heavy skewness through `log1p` transformation of:
  - `amount_tsh`
  - `amount_tsh_capped`
  - `water_risk_score`
  - `pump_age`
- Created new columns with `_log` suffix to preserve the original data.
- Detected that `num_private` was extremely sparse (mostly 0), and therefore:
  - Replaced it with a new binary feature `has_private_owner` (1 if private ownership exists, 0 otherwise).

### 📦 Data Saving
- Saved the updated dataset after feature engineering as `feature_engineered_data_V1.csv`.
- This version will be used as the basis for further feature selection and modeling.

## 🔍 Numerical Feature Correlation Analysis

We performed Spearman correlation analysis to identify potential redundancy among numerical features. The goal was to reduce multicollinearity and keep only the most informative features.

### ✅ Observations:


- **`construction_year`** and **`pump_age`** had a perfect negative correlation (-1.00).  
  → We kept `pump_age` and dropped `construction_year` as it is easier to interpret and more useful for modeling.

- **`amount_tsh_log`**, **`amount_tsh_capped_log`**, and **`water_risk_score_log`** were all perfectly correlated (~1.00).  
  → We retained only `water_risk_score_log` to avoid redundancy.

- Features like **`gps_height`**, **`latitude`**, **`longitude`**, **`population_log`**, and **`has_private_owner`** showed low correlation with others and were retained for modeling.
- selected columns to drop: `num_private`,`construction_year`,`amount_tsh_log`, `amount_tsh_capped_log`,


###  Categorical Correlation (Cramér's V)

We evaluated categorical feature relationships after cleaning and correcting data types. No pairs showed dangerously high redundancy (Cramér’s V > 0.7), so we retained all final features. Examples of moderate but useful associations included:

- `payment` and `water_risk_score_log`: 0.54
- `pump_age_binned` and `region`: 0.46
- `quantity` and `status_group`: 0.31

These relationships were considered **informative rather than redundant**, and the features were retained for modeling.

## 🔍 Logistic Regression Baseline Model

We trained a baseline **multiclass logistic regression** model to predict pump functionality (`functional`, `non functional`, `functional needs repair`). 

### ✅ Preprocessing Steps

- **Target encoding**: `status_group` was label-encoded into numeric classes.
- **Feature scaling**: All features were standardized using `StandardScaler`.
- **Target leakage prevention**: One-hot encoded `status_group_*` columns and the unique `id` column were removed from the training features.

### ⚙️ Cross-Validation Results (5-Fold)

```text
Fold scores: [0.730, 0.728, 0.724, 0.724, 0.733]
Mean accuracy: **72.78%**

##  Best Logistic Regression Model

After tuning hyperparameters using `GridSearchCV`, we saved the best-performing model — including preprocessing (scaling) — using `joblib`. This makes it easy to reuse the trained model without retraining from scratch.

### Best Model Configuration

The selected model was:

```text
Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(C=0.01, max_iter=1000, solver='saga'))
])