from pathlib import Path
import pandas as pd

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
CURRENT_ID = "SK_ID_CURR"
PREV_ID = "SK_ID_PREV"
BUREAU_ID = "SK_ID_BUREAU"

# Just for exploration purposes
SAMPLE_ID_CURR = 350419

TARGET = "TARGET"

# Util functions
def add_count(df, aux_df, aux_id, new_colname, subset=None, main_id=CURRENT_ID):
    """Returns new df"""
    if subset is not None:
        aux_df = aux_df.loc[subset]
    df_ext = df.merge(aux_df, on=main_id, how="left")
    counts = df_ext.groupby(main_id)[aux_id].count().reset_index()
    counts.rename(columns={aux_id: new_colname}, inplace=True)
    df = df.merge(counts, on=main_id, how="left")
    df[new_colname].fillna(0, inplace=True)
    return df


def add_exposure(df, aux_df, subset, exp_col, new_col, main_id=CURRENT_ID):
    aux_df = aux_df.loc[subset].copy()
    # if na in exposure column, assume 0
    aux_df[exp_col].fillna(0, inplace=True)
    df_ext = df.merge(aux_df, on=main_id, how="left")
    exposure = df_ext.groupby(main_id)[exp_col].sum().reset_index()
    exposure.rename(columns={exp_col: new_col}, inplace=True)
    return df.merge(exposure, on=main_id, how="left")


def add_age_credit(df, aux_df, age_col, new_colname, type, na_imp=None, main_id=CURRENT_ID):
    """type can be either 'last' for most recent, 'age' for age of the oldest loan"""
    assert type in ["last", "age"], "type must be either 'last' or 'age'"
    agg_type = "max" if type == "last" else "min"
    df_ext = df.merge(aux_df, on=main_id, how="left")
    last_credit = df_ext.groupby(main_id)[age_col].agg(agg_type).reset_index()
    last_credit.rename(columns={age_col: new_colname}, inplace=True)
    df = df.merge(last_credit, on=main_id, how="left")
    if na_imp is not None:
        df[new_colname].fillna(na_imp, inplace=True)
    return df


def load_train_test(only_train=False):
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
    X_train = train.drop(columns=[TARGET, CURRENT_ID])
    y_train = train[TARGET]

    if not only_train:
        test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")
        X_test = test.drop(columns=[TARGET, CURRENT_ID])
        y_test = test[TARGET]
        return train, X_train, y_train, test, X_test, y_test

    return train, X_train, y_train