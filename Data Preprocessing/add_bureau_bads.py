import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, BUREAU_ID, add_count

from pdb import set_trace
import gc

# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
# keep memory lean
gc.collect()

# Credit Bureau
bureau = pd.read_csv(PARENT_DIR / "bureau.csv")
bureau = bureau[[CURRENT_ID, BUREAU_ID, "CREDIT_DAY_OVERDUE"]]
gc.collect()

# Feature 1 - Number of Bureau Credits overdue >= 30 days
df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_CURR_BAD_30", bureau.CREDIT_DAY_OVERDUE >= 30)

# Feature 2 - Number of bureau credits overdue >= 60 days
df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_CURR_BAD_60", bureau.CREDIT_DAY_OVERDUE >= 60)

set_trace()