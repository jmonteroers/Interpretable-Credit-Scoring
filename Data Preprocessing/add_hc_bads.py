"""This script will use Home Credit Balances up to 12 months old to count the number of HC loans going bad in these windows. 
Using SK_DPD_DEF attribute to signal delayed payments"""

import pandas as pd
from utils import PARENT_DIR, PREV_ID, add_count

from pdb import set_trace
import gc

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


