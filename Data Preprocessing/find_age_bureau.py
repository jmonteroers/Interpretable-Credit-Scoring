"""
First example of function to engineer features, to be extended!
"""

import pandas as pd
from utils import PARENT_DIR, CURRENT_ID

from pdb import set_trace


def find_age_bureau(df, bureau):
    ext_df = df.merge(bureau, how="left", on=CURRENT_ID)
    # Note: DAYS_CREDIT is relative to application date, negative
    age_bureau = ext_df.groupby(CURRENT_ID).agg({"DAYS_CREDIT": "min"}).reset_index()
    age_bureau.rename(columns={"DAYS_CREDIT": "AGE_BUREAU"}, inplace=True)
    # merge back to original df
    df = df.merge(age_bureau, how="left", on=CURRENT_ID)
    return df


def find_age_prev_apps(df, prev_apps):
    """I proxy the age using the maximum days since decision on previous application"""
    ext_df = df.merge(prev_apps, how="left", on=CURRENT_ID)
    age_prev_apps = ext_df.groupby(CURRENT_ID).agg({"DAYS_DECISION": "min"}).reset_index()
    age_prev_apps.rename(columns={"DAYS_DECISION": "AGE_PREV"}, inplace=True)
    df = df.merge(age_prev_apps, how="left", on=CURRENT_ID)
    return df


if __name__ == "__main__":
    # only working with train data - NOTE: Currently sample only!
    df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
    bureau = pd.read_csv(PARENT_DIR / "bureau.csv")
    prev_apps = pd.read_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")
    df_with_age_bureau = find_age_bureau(df, bureau)
    fed_df = find_age_prev_apps(df_with_age_bureau, prev_apps)

    set_trace()
    # fed_df.to_csv(PARENT_DIR / 'processed' / 'train_raw_apps.csv.zip')
