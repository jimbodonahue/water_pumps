# Using the cleaning function on the original training data

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, 'data')
out_path = os.path.join(parent_dir, 'outputs')

# Read the files
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
print(train.columns)

cleaner = DataCleaner()
X = cleaner.clean_data(train)
print("\nCleaned DataFrame:")
print(X.head())#_small.head())
