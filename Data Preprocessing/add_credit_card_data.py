import pandas as pd
from utils import PARENT_DIR, CURRENT_ID

from pdb import set_trace
import gc

# Load datasets
df = pd.read_csv(PARENT_DIR / 'processed' / 'sample_train_raw_apps.csv.zip', compression="zip")
cc_balance = pd.read_csv(PARENT_DIR / "credit_card_balance.csv.zip", compression="zip")
# only keep required columns
cc_balance = cc_balance[[CURRENT_ID, "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]]
cc_balance["MONTHS_BALANCE"] = pd.to_numeric(cc_balance["MONTHS_BALANCE"], downcast="integer")
# Join datasets
df_ext = df.merge(cc_balance, how="left", on=CURRENT_ID)

# For experiments, only keep one client
df_ext = df_ext.loc[df_ext.SK_ID_CURR == SAMPLE_ID_CURR, ]
df_ext.MONTHS_BALANCE = df_ext.MONTHS_BALANCE.astype('int8')
del(cc_balance)
gc.collect()

# Feature 1 - Active credit card
def has_active_credit_card(df_g):
    """Apply-type function. Only look at last month, if AMT_BALANCE 0 or NaN, return 1, otherwise 0"""
    df_g = df_g.loc[df_g.MONTHS_BALANCE == -1, ]
    balance_last_month = df_g.iloc[0, df_g.columns.get_loc("AMT_BALANCE")]
    return pd.Series({"ACTIVE_CREDIT_CARD": int(not pd.isna(balance_last_month) and balance_last_month > 1e-8)})

df_f1 = df_ext.groupby(CURRENT_ID).apply(has_active_credit_card).reset_index()

# Feature 2 - Over the limit, current and last three months
def has_credit_over_limit(df_g):
    """Apply-type function. Only look at last three months, 
    if AMT_BALANCE > AMT_CREDIT_LIMIT_ACTUAL any month, return 1, otherwise 0. 
    Return for last month (_curr) and last three months (_tm)"""
    df_g = df_g.loc[df_g.MONTHS_BALANCE >= -3, ["MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]]
    # if not three months, then NA
    over_limit_tm = pd.NA
    if (len(df_g) == 3):
        over_limit_tm = int((df_g.AMT_BALANCE > df_g.AMT_CREDIT_LIMIT_ACTUAL).any())
    # if not one month, then NA
    over_limit_curr = pd.NA
    if (len(df_g) >= 1):
        # just keep last month
        df_g = df_g.loc[df_g.MONTHS_BALANCE == -1, ]
        over_limit_curr = int((df_g.AMT_BALANCE > df_g.AMT_CREDIT_LIMIT_ACTUAL).any())
    return pd.Series({"CC_OVERLIM_QRT": over_limit_tm, "CC_OVERLIM_CURR": over_limit_curr})

df_f2 = df_ext.groupby(CURRENT_ID).apply(has_credit_over_limit).reset_index()

# Feature 3 - Average Consumption, last three months
def avg_perc_cons_qrt(df_g):
    """Apply-type function. Only look at last three months, 
    return average percentage of consumption of credit limit"""
    df_g = df_g.loc[df_g.MONTHS_BALANCE >= -3, ["MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]]
    avg_perc_cons = pd.NA
    if (len(df_g) == 3):
        avg_perc_cons = (df_g["AMT_BALANCE"] / df_g["AMT_CREDIT_LIMIT_ACTUAL"]).mean()
    return pd.Series({"CC_AVG_PERC_CONS_QRT": avg_perc_cons})

df_f3 = df_ext.groupby(CURRENT_ID).apply(avg_perc_cons_qrt).reset_index()
set_trace()
