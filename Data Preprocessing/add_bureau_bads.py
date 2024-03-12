import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, BUREAU_ID, PREV_ID

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

def add_count(df, aux_df, aux_id, subset, new_colname, main_id=CURRENT_ID):
    """Returns new df"""
    aux_df = aux_df.loc[subset]
    df_ext = df.merge(aux_df, on=main_id, how="left")
    counts = df_ext.groupby(main_id)[aux_id].count().reset_index()
    counts.rename(columns={aux_id: new_colname}, inplace=True)
    df = df.merge(counts, on=main_id, how="left")
    df[new_colname].fillna(0, inplace=True)
    return df

# Feature 1 - Number of Bureau Credits overdue >= 30 days
df = add_count(df, bureau, BUREAU_ID, bureau.CREDIT_DAY_OVERDUE >= 30, "N_BUREAU_CURR_BAD_30")

# Feature 2 - Number of bureau credits overdue >= 60 days
df = add_count(df, bureau, BUREAU_ID, bureau.CREDIT_DAY_OVERDUE >= 60, "N_BUREAU_CURR_BAD_60")

set_trace()