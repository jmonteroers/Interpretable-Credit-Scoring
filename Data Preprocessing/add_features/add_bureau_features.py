import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, BUREAU_ID, add_count, add_age_credit, add_exposure

import gc

# TODO: For ref, add minimum age in train to main

def add_bureau_features(df, parent_dir=PARENT_DIR):
    # Credit Bureau
    bureau = pd.read_csv(parent_dir / "bureau.csv")
    bureau = bureau[[CURRENT_ID, BUREAU_ID, "DAYS_CREDIT", "CREDIT_ACTIVE", "AMT_CREDIT_SUM", "CREDIT_DAY_OVERDUE"]]
    # keep memory lean
    gc.collect()
    # Feature - Number of active bureau bureau credits
    df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_ACTIVE", bureau.CREDIT_ACTIVE == "Active")

    # Feature - Number of bureau credits contracted last year
    df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_LAST_YEAR", bureau.DAYS_CREDIT >= -365.)

    # Feature - Time since last bureau credit
    # nas imputed using minimum in train
    df = add_age_credit(df, bureau, "DAYS_CREDIT", "AGE_LAST_BUREAU", type="last", na_imp=-2922)

    # Feature - Bureau Age
    # If not available, imputed as 0
    df = add_age_credit(df, bureau, "DAYS_CREDIT", "AGE_BUREAU", type="age", na_imp=0)

    # Feature - Total Exposure, Active Bureau Credits
    # TODO: Build total exposure in add_features
    df = add_exposure(df, bureau, bureau.CREDIT_ACTIVE == "Active", "AMT_CREDIT_SUM", "BUREAU_EXP")

    # Feature - Number of Bureau Credits overdue >= 30 days
    df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_CURR_BAD_30", bureau.CREDIT_DAY_OVERDUE >= 30)

    # Feature - Number of bureau credits overdue >= 60 days
    df = add_count(df, bureau, BUREAU_ID, "N_BUREAU_CURR_BAD_60", bureau.CREDIT_DAY_OVERDUE >= 60)

    return df


