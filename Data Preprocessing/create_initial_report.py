import pandas as pd
from utils import PARENT_DIR

from pdb import set_trace

df = pd.read_csv(PARENT_DIR / "processed" / "train_raw_apps.csv.zip", compression="zip")

# Assuming your DataFrame is named df
summary_data = pd.DataFrame(index=df.columns)

# Calculate unique values, mode, dtype, and number of missing values for each column
summary_data['Unique Values'] = df.nunique()
summary_data['Mode'] = df.mode().iloc[0]
summary_data['Dtype'] = df.dtypes
summary_data['% Missing Values'] = 100 * df.isnull().mean()

# Extend the DataFrame to include min, max, median, and mean for numeric columns
numeric_summary = df.describe().transpose()
summary_data[['Min', 'Max', 'Median', 'Mean']] = numeric_summary[['min', 'max', '50%', 'mean']]

summary_data.to_excel(PARENT_DIR / "train_summary_applications.xlsx")

set_trace()
