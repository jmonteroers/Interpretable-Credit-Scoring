"""
This script finds the IV of each variable, then takes as input the pre-processed dataset
and then filters out any variable with IV lower than 0.02 (indicating it to be generally unpredictive, Siddiqi, 2017)
"""

import pandas as pd
import logging

from utils.utils import PARENT_DIR, TARGET
from utils.attrs import prettify_attrs

logger = logging.getLogger("WOE_logger")
logging.basicConfig(level=logging.INFO)

woe_res = pd.read_excel(PARENT_DIR / "meta" / "woe_map" / "woe_mapping.xlsx")
# select rows referring to Totals
woe_totals = woe_res.loc[woe_res.Bin.isna()]

logger.info(f"Number of Totals selected: {len(woe_totals)}")

# Pick Attribute and Total IV columns
iv = woe_totals.loc[:, ["Attribute", "IV"]]
iv.sort_values("IV", ascending=False, inplace=True)

# Follow Siddiqi (2017), apply IV filter for IV less than 0.02 (generally unpredictive)
iv_sel = iv.loc[iv.IV >= 0.02].copy()
logger.info(f"After IV filtering, {len(iv_sel)} features have been selected")

# Export to LaTeX table
iv_out = iv_sel.copy()
outpath_table = PARENT_DIR / "meta" / "iv_table.tex"

# Create table by Strength
def iv_to_strength(iv):
    if iv < 0.02:
        return "Generally Unpredictive"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    return "Strong"

iv_out["Strength"] = iv_out["IV"].apply(iv_to_strength)
# Map attributes to clean attributes
iv_out = prettify_attrs(iv_out, 'Attribute')
iv_by_strength = iv_out.groupby('Strength')['Attribute'].agg(', '.join).reset_index()
iv_by_strength.style.\
       format(escape="latex").\
       hide(axis="index").\
       to_latex(outpath_table, hrules=True)

# Apply filtering
# load dataset
from utils.utils import PARENT_DIR
train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
# filter
sel_attributes = iv_sel.Attribute.unique().tolist()
train = train[["SK_ID_CURR", TARGET] + sel_attributes]
# save
train.to_csv(PARENT_DIR / "processed" / "train_apps_iv.csv.zip", index=False)

breakpoint()
