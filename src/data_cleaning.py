
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd # geospatial data processing
import matplotlib.pyplot as plt # standard plotting packages
import seaborn as sns
import os

# Build path to data folder
path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Read the files
train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
labels = pd.read_csv(os.path.join(path, 'train_labels.csv'))

# Build path to data folder
path = os.path.join(os.path.dirname(__file__), '..', 'data')

# Read the files
train = pd.read_csv(os.path.join(path, 'train.csv'))
test = pd.read_csv(os.path.join(path, 'test.csv'))
labels = pd.read_csv(os.path.join(path, 'train_labels.csv'))

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Labels shape:", labels.shape)

print("\nTrain columns:")
print(train.columns)

#Initial exploring, missing values
