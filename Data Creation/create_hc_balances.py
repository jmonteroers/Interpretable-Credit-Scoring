"""
Takes Home Credit Balances as input (Credit Card, POS), selects some columns (`SELECTED_COLS`) and concatenate 
them vertically to produce a single Home Credit Balance dataset (prev_balances)
"""

import pandas as pd
from pathlib import Path

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
CURRENT_ID = "SK_ID_CURR"
SELECTED_COLS = [CURRENT_ID, "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"]

# previous home credit loan balances, only keep last snapshot, relevant columns.
pos_balances = pd.read_csv(PARENT_DIR / "POS_CASH_balance.csv.zip", compression="zip")
pos_balances = pos_balances[SELECTED_COLS]
cc_balances = pd.read_csv(PARENT_DIR / "credit_card_balance.csv.zip", compression="zip")
cc_balances = cc_balances[SELECTED_COLS]
prev_balances = pd.concat([pos_balances, cc_balances], axis=0)
prev_balances.to_csv(PARENT_DIR / "processed" / "prev_balances.csv.zip", index=False)