"""
This script takes as input the data after main preprocessing steps (feature engineering, imputation, etc)
and applies WOE to either only categorical predictors or both categorical and numerical predictors. To distinguish between
numerical and categorical predictors, it utilises the metadata in 'train_summary_applications_ext.csv'
"""

import pandas as pd
from optbinning import OptimalBinning
import time

from add_features.utils import PARENT_DIR

TARGET = "TARGET"


class Timer(object):
    "Borrowed from @Bendersky, stackoverflow"
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def fit_woe_transform(train: pd.DataFrame, pred_name: str, type: str):
    "type is either 'numerical' or 'categorical'. Returns fitted binning object"
    x = train[pred_name].values
    y = train.TARGET.values

    if type == "categorical":
        optb = OptimalBinning(
                name=pred_name, dtype="categorical", solver="mip", cat_cutoff=0.1
        )
    else:
        optb = OptimalBinning(name=pred_name, dtype="numerical", solver="cp")

    optb.fit(x, y)

    # check the status
    if optb.status != "OPTIMAL":
        print(f"Warning: WoE Binning for {pred_name} has not converged. Status thrown: {optb.status}")

    return optb


# Load datasets
train = pd.read_csv(PARENT_DIR / "processed" / "train_apps_imp.csv.gz", compression="gzip")
test  = pd.read_csv(PARENT_DIR / "processed" / "test_apps_imp.csv.gz", compression="gzip")

# Metadata
meta = pd.read_csv(PARENT_DIR / "meta" / "train_summary_applications_ext.csv")

# Get categorical columns
cat_columns = meta.loc[meta["Data Type"].isin(["Categorical", "Binary"]), "Attribute"].unique().tolist()
num_columns = meta.loc[meta["Data Type"].isin(["Quantitative"]), "Attribute"].unique().tolist()

# Transform into woe - non-parallel
with Timer("Non parallel WoE for Categorical"):
    for cat in cat_columns:
        cat_optb = fit_woe_transform(train, cat, "categorical")
        train[cat] = cat_optb.transform(train[cat], metric="woe")
        test[cat] = cat_optb.transform(test[cat], metric="woe")

with Timer("Non parallel WoE for Numerical"):
    for num in num_columns:
        num_optb = fit_woe_transform(train, num, "numerical")
        train[num] = num_optb.transform(train[num], metric="woe")
        test[num] = num_optb.transform(test[num], metric="woe")


breakpoint()