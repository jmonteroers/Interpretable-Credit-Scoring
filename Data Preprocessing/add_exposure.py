import pandas as pd
from utils import PARENT_DIR, CURRENT_ID

from pdb import set_trace
import gc

# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
# keep memory lean
gc.collect()

def add_exposure(df, aux_df, subset, exp_col, new_col, main_id=CURRENT_ID):
    aux_df = aux_df.loc[subset].copy()
    # if na in exposure column, assume 0
    aux_df[exp_col].fillna(0, inplace=True)
    df_ext = df.merge(aux_df, on=main_id, how="left")
    exposure = df_ext.groupby(main_id)[exp_col].sum().reset_index()
    exposure.rename(columns={exp_col: new_col}, inplace=True)
    return df.merge(exposure, on=main_id, how="left")


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

