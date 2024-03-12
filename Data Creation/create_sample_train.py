import pandas as pd
from pathlib import Path

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'

df = pd.read_csv(PARENT_DIR / 'processed' / 'train_raw_apps.csv.zip', compression="zip")

sample_df = df.sample(n = 10000)

df.to_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', index=False)