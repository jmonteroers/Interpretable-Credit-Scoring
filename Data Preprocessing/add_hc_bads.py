"""This script will use Home Credit Balances up to 12 months old to count the number of HC loans going bad in these windows. 
Using SK_DPD_DEF attribute to signal delayed payments"""

import pandas as pd
from pathlib import Path

from pdb import set_trace
import gc

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
CURRENT_ID = "SK_ID_CURR"
PREV_ID = "SK_ID_PREV"
BUREAU_ID = "SK_ID_BUREAU"

def add_count(df, aux_df, aux_id, new_colname, subset=None, main_id=CURRENT_ID):
    """Returns new df"""
    if subset is not None:
        aux_df = aux_df.loc[subset]
    df_ext = df.merge(aux_df, on=main_id, how="left")
    counts = df_ext.groupby(main_id)[aux_id].count().reset_index()
    counts.rename(columns={aux_id: new_colname}, inplace=True)
    df = df.merge(counts, on=main_id, how="left")
    df[new_colname].fillna(0, inplace=True)
    return df


# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
prev_balances = pd.read_csv(PARENT_DIR / 'processed' / 'prev_balances.csv.zip', compression="zip")
# only keep last 12 months
prev_balances = prev_balances.loc[prev_balances.MONTHS_BALANCE >= -12.]
# keep memory lean
gc.collect()


# Feature 1 - count number of bad loans (>30) just looking at last months balance
df = add_count(df, prev_balances, PREV_ID, "N_HC_BAD_30_CURR", (prev_balances.MONTHS_BALANCE >= -1.) & (prev_balances.SK_DPD_DEF >= 30.))

# Feature 2 - count number of bad loans (>30) looking at last quarter balance
df = add_count(df, prev_balances, PREV_ID, "N_HC_BAD_30_QRT", (prev_balances.MONTHS_BALANCE >= -3.) & (prev_balances.SK_DPD_DEF >= 30.))

# Feature 3 - count number of bad loans (>30) looking at last year balance
df = add_count(df, prev_balances, PREV_ID, "N_HC_BAD_30_YR", prev_balances.SK_DPD_DEF >= 30.)

set_trace()


