import pandas as pd
from utils import PARENT_DIR, CURRENT_ID, PREV_ID, add_count, add_age_credit, add_exposure

import gc

# TODO: For ref, add minimum age in train to main

def add_prev_apps_features(df, parent_dir=PARENT_DIR):
    # Load aux dataset
    # NOTE: Only loading approved previous applications
    prev_apps = pd.read_csv(parent_dir / "processed" / "approved_previous_application.csv.zip")
    prev_apps = prev_apps[[CURRENT_ID, PREV_ID, "DAYS_DECISION", "DAYS_TERMINATION", "AMT_CREDIT"]]
    # to avoid issues when merging with main
    prev_apps.rename(columns={"AMT_CREDIT": "AMT_CREDIT_PREV"}, inplace=True)
    gc.collect()
    # Feature - Number of active previous applications
    df = add_count(df, prev_apps, PREV_ID, "N_PREV_ACTIVE", prev_apps.DAYS_TERMINATION >= 0)

    # Feature - Number of previous Home credits last year
    df = add_count(df, prev_apps, PREV_ID, "N_PREV_LAST_YEAR", prev_apps.DAYS_DECISION >= -365.)

    # Feature - Time since last previous HC credit
    df = add_age_credit(df, prev_apps, "DAYS_DECISION", "AGE_LAST_HC", type="last", na_imp=-2922)

    # Feature - Age Oldest approved HC application
    df = add_age_credit(df, prev_apps, "DAYS_DECISION", "AGE_HC", type="age", na_imp=0)

    # Feature - Existing HC Total Exposure
    df = add_exposure(df, prev_apps, prev_apps.DAYS_TERMINATION >= 0, "AMT_CREDIT_PREV", "HC_EXP")
    
    return df