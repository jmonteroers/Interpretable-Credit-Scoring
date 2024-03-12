import pandas as pd
from utils import PARENT_DIR

df = pd.read_csv(PARENT_DIR / 'processed' / 'train_raw_apps.csv.zip', compression="zip")
sample_df = df.sample(n = 10000, random_state=1234)
df.to_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', index=False)