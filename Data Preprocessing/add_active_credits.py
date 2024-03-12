import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, BUREAU_ID, PREV_ID, add_count, add_age_credit

from pdb import set_trace
import gc

# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
# keep memory lean
gc.collect()

# Credit Bureau
bureau = pd.read_csv(PARENT_DIR / "bureau.csv")
bureau = bureau[[CURRENT_ID, BUREAU_ID, "DAYS_CREDIT", "CREDIT_ACTIVE"]]
gc.collect()

# Feature 1 - Number of active credit bureau
df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_ACTIVE", bureau.CREDIT_ACTIVE == "Active")

# Feature 2 - Number of bureau credits contracted last year
df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_LAST_YEAR", bureau.DAYS_CREDIT >= -365.)

# Feature 3 - Time since last bureau credit
# nas imputed using minimum in train
df = add_age_credit(df, bureau, "DAYS_CREDIT", "AGE_LAST_BUREAU", type="last", na_imp=-2922)

# Previous Applications
del(bureau)
prev_apps = pd.read_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")
prev_apps = prev_apps[[CURRENT_ID, PREV_ID, "DAYS_DECISION", "DAYS_TERMINATION"]]
gc.collect()

# Feature 4 - Number of active previous applications
df = add_count(df, prev_apps, PREV_ID, "N_PREV_ACTIVE", prev_apps.DAYS_TERMINATION >= 0)

# Feature 5 - Number of previous Home credits last year
df = add_count(df, prev_apps, PREV_ID, "N_PREV_LAST_YEAR", prev_apps.DAYS_DECISION >= -365.)

# Feature 6 - Time since last previous HC credit
df = add_age_credit(df, prev_apps, "DAYS_DECISION", "AGE_LAST_HC", type="last", na_imp=-2922)

print(df.iloc[:5, -6:])

set_trace()