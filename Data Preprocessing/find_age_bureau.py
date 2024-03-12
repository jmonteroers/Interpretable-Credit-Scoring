"""
Add age in bureau and HC
"""

import pandas as pd
from utils import PARENT_DIR, add_age_credit

from pdb import set_trace

if __name__ == "__main__":
    # only working with train data - NOTE: Currently sample only!
    df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
    bureau = pd.read_csv(PARENT_DIR / "bureau.csv")
    prev_apps = pd.read_csv(PARENT_DIR / "processed" / "approved_previous_application.csv.zip")


    df_with_age_bureau = add_age_credit(df, bureau, "DAYS_CREDIT", "AGE_BUREAU", type="age", na_imp=0)
    df_with_age_prev = add_age_credit(df, prev_apps, "DAYS_DECISION", "AGE_HC", type="age", na_imp=0)

    set_trace()
    # fed_df.to_csv(PARENT_DIR / 'processed' / 'train_raw_apps.csv.zip')
