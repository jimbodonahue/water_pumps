import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load cleaned base data from data_cleaning.py
path = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(path,'cleaned_data_filled_V5.csv'))
print(df[['construction_year', 'gps_height', 'longitude']].isnull().sum())