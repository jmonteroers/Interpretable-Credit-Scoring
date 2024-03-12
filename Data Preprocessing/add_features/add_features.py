import pandas as pd

from utils import PARENT_DIR

from add_bureau_features import add_bureau_features
from add_prev_apps_features import add_prev_apps_features
from add_hc_balance_features import add_hc_balance_feats
from add_credit_card_features import add_cc_features


def add_features(parent_dir=PARENT_DIR):
    # Load applications
    df = pd.read_csv(parent_dir / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")

    df = add_bureau_features(df, parent_dir)
    df = add_prev_apps_features(df, parent_dir)
    df = add_hc_balance_feats(df, parent_dir)
    df = add_cc_features(df, parent_dir)

    # TODO: Add total exposure
    return df


if __name__ == "__main__":
    from datetime import datetime
    from pdb import set_trace

    print(f"Starting to add features: {datetime.now()}")
    df_check = add_features()
    print(f"Completed - add features: {datetime.now()}")
    print(df_check.iloc[0, -20:])
    set_trace()