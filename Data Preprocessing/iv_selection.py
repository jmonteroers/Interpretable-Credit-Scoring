"""
This script finds the IV of each variable, and then takes as input the pre-processed dataset
and then filters out any variable with IV lower than 0.02 (indicating it to be generally unpredictive, Siddiqi, 2017)
"""

import pandas as pd
import logging

from add_features.utils import PARENT_DIR

logger = logging.getLogger("WOE_logger")
logging.basicConfig(level=logging.INFO)

TARGET = "TARGET"

woe_res = pd.read_csv(PARENT_DIR / "meta" / "woe_mapping.csv.zip")
# select rows referring to Totals
woe_totals = woe_res.loc[woe_res.Bin.isna()]

logger.info(f"Number of Totals selected: {len(woe_totals)}")

# Pick Attribute and Total IV columns
iv = woe_totals.loc[:, ["Attribute", "IV"]]
iv.sort_values("IV", ascending=False, inplace=True)

# Follow Siddiqi (2017), apply IV filter for IV less than 0.02 (generally unpredictive)
iv_sel = iv.loc[iv.IV >= 0.02]
logger.info(f"After IV filtering, {len(iv_sel)} have been selected")

# Export to LaTeX table
outpath_table = PARENT_DIR / "meta" / "iv_table.tex"
iv_sel.style.\
       format(escape="latex").\
       format(subset="IV", precision=2).\
       hide(axis="index").\
       to_latex(outpath_table, hrules=True)

# TODO: Apply filtering

breakpoint()
