import pandas as pd
import numpy as np
from utils.utils import PARENT_DIR, CURRENT_ID, PREV_ID, SAMPLE_ID_CURR, add_count

import gc

EPS = 1e-8

# Average Consumption - Util Functions
def apply_avg_prop_cons(df_g, out_col):
    """Apply-type function. Return average proportion of credit limit consumption"""
    avg_perc_cons = (df_g["AMT_BALANCE"] / df_g["AMT_CREDIT_LIMIT_ACTUAL"]).mean()
    return pd.Series({out_col: avg_perc_cons})

def add_prop_cons(df, aux_df, cons_col, new_col, subset=None, main_id=CURRENT_ID):
    if subset is not None:
        aux_df = aux_df.loc[subset]
    df_ext = df.merge(aux_df, on=main_id, how="left")
    consumption = df_ext.groupby(CURRENT_ID)[cons_col].mean().reset_index()
    consumption.rename(columns={cons_col: new_col}, inplace=True)
    return df.merge(consumption, on=main_id, how="left")


def add_cc_features(df, parent_dir=PARENT_DIR):
    # Load aux dataset
    cc_balance = pd.read_csv(parent_dir / "credit_card_balance.csv.zip", compression="zip")

    # only keep required columns
    cc_balance = cc_balance[[CURRENT_ID, PREV_ID, "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]]
    cc_balance.MONTHS_BALANCE = cc_balance.MONTHS_BALANCE.astype('int8')
    # define proportion of credit limit consumption
    cc_balance["PROP_CREDIT_LIMIT_CONS"] = cc_balance["AMT_BALANCE"] / cc_balance["AMT_CREDIT_LIMIT_ACTUAL"]
    # fix infinity values for proportion (when no credit limit available)
    median_prop = cc_balance["PROP_CREDIT_LIMIT_CONS"].median()
    cc_balance["PROP_CREDIT_LIMIT_CONS"].replace([np.inf, -np.inf], np.nan, inplace=True)
    cc_balance["PROP_CREDIT_LIMIT_CONS"].fillna(median_prop, inplace=True)

    # only using last 3 months of data
    cc_balance = cc_balance.loc[cc_balance.MONTHS_BALANCE >= -3]
    # keep memory lean
    gc.collect()

    # Feature - Add Count HC Active Credit Cards
    df = add_count(
        df, cc_balance, PREV_ID, "COUNT_ACTIVE_CC_CURR", 
        (cc_balance.MONTHS_BALANCE == -1) & (cc_balance.AMT_BALANCE > EPS)
        )

    # Feature - Over the limit, current
    df = add_count(
        df, cc_balance, PREV_ID, "COUNT_CC_OVER_LIMIT_CURR", 
        (cc_balance.MONTHS_BALANCE == -1) & (cc_balance.AMT_BALANCE > cc_balance.AMT_CREDIT_LIMIT_ACTUAL)
    )

    # Feature - Times over the limit, last three months
    df = add_count(
        df, cc_balance, PREV_ID, "TIMES_CC_OVER_LIMIT_QRT", (cc_balance.AMT_BALANCE > cc_balance.AMT_CREDIT_LIMIT_ACTUAL)
    )

    # Feature - Last Quarter proportion consumption
    df = add_prop_cons(df, cc_balance, "PROP_CREDIT_LIMIT_CONS", "CC_PROP_CONS_CURR", subset=cc_balance.MONTHS_BALANCE == -1.)

    # Feature - Current proportion consumption, looking at last month
    df = add_prop_cons(df, cc_balance, "PROP_CREDIT_LIMIT_CONS", "CC_PROP_CONS_QRT")

    return df
