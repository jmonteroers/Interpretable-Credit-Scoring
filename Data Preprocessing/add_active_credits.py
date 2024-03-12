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
bureau = bureau[[CURRENT_ID, BUREAU_ID, "DAYS_CREDIT", "CREDIT_ACTIVE"]]
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


def add_last_credit(df, aux_df, age_col, new_colname, na_imp=None, main_id=CURRENT_ID):
    """Returns new df"""
    df_ext = df.merge(aux_df, on=main_id, how="left")
    last_credit = df_ext.groupby(main_id)[age_col].max().reset_index()
    last_credit.rename(columns={age_col: new_colname}, inplace=True)
    df = df.merge(last_credit, on=main_id, how="left")
    if na_imp is not None:
        df[new_colname].fillna(na_imp, inplace=True)
    return df

# Feature 1 - Number of active credit bureau
df = add_count(df, bureau, BUREAU_ID, bureau.CREDIT_ACTIVE == "Active", "N_BUREAU_ACTIVE")

# Feature 2 - Number of bureau credits contracted last year
df = add_count(df, bureau, BUREAU_ID, bureau.DAYS_CREDIT >= -365., "N_BUREAU_LAST_YEAR")

# Feature 3 - Time since last bureau credit
# nas imputed using minimum in train
df = add_last_credit(df, bureau, "DAYS_CREDIT", "AGE_LAST_BUREAU", na_imp=-2922)

# Previous Applications
del(bureau)
prev_apps = pd.read_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")
prev_apps = prev_apps[[CURRENT_ID, PREV_ID, "DAYS_DECISION", "DAYS_TERMINATION"]]
gc.collect()

# Feature 4 - Number of active previous applications
df = add_count(df, prev_apps, PREV_ID, prev_apps.DAYS_TERMINATION >= 0, "N_PREV_ACTIVE")

# Feature 5 - Number of previous Home credits last year
df = add_count(df, prev_apps, PREV_ID, prev_apps.DAYS_DECISION >= -365., "N_PREV_LAST_YEAR")

# Feature 6 - Time since last previous HC credit
df = add_last_credit(df, prev_apps, "DAYS_DECISION", "AGE_LAST_HC", na_imp=-2922)

print(df.iloc[:5, -6:])

set_trace()