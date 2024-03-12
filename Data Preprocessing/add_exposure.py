import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, add_exposure

from pdb import set_trace
import gc

# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
# keep memory lean
gc.collect()


# Credit Bureau
bureau = pd.read_csv(PARENT_DIR / "bureau.csv")
bureau = bureau[[CURRENT_ID, "CREDIT_ACTIVE", "AMT_CREDIT_SUM"]]
gc.collect()

df = add_exposure(df, bureau, bureau.CREDIT_ACTIVE == "Active", "AMT_CREDIT_SUM", "BUREAU_EXP")

# Previous applications
prev_apps = pd.read_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")
prev_apps = prev_apps[[CURRENT_ID, "DAYS_TERMINATION", "AMT_CREDIT"]]
# to avoid issues when merging with main
prev_apps.rename(columns={"AMT_CREDIT": "AMT_CREDIT_PREV"}, inplace=True)
gc.collect()

df = add_exposure(df, prev_apps, prev_apps.DAYS_TERMINATION >= 0, "AMT_CREDIT_PREV", "HC_EXP")

# Total exposure
df["EXPOSURE_PERC_INC"] = (df["BUREAU_EXP"] + df["HC_EXP"]) / df["AMT_INCOME_TOTAL"]
# Remove intermediate variables
df.drop(columns=["BUREAU_EXP", "HC_EXP"], inplace=True)
print(df.iloc[:5, -1])

