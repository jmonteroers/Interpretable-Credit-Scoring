"""
This script takes as input the data after main preprocessing steps (feature engineering, imputation, etc)
and applies WOE to either only categorical predictors or both categorical and numerical predictors. To distinguish between
numerical and categorical predictors, it utilises the metadata in 'train_summary_applications_ext.csv'
"""

import pandas as pd
from optbinning import OptimalBinning
import time
import logging

from add_features.utils import PARENT_DIR

logger = logging.getLogger("WOE_logger")
logging.basicConfig(level=logging.INFO)

TARGET = "TARGET"
RANDOM_SEED = 1234

class Timer(object):
    "Borrowed from @Bendersky, stackoverflow"
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        logger.info('Elapsed: %s' % (time.time() - self.tstart))


def fit_woe_transform(train: pd.DataFrame, pred_name: str, type: str):
    "type is either 'numerical' or 'categorical'. Returns fitted binning object"
    x = train[pred_name].values
    y = train.TARGET.values

    if type == "categorical":
        optb = OptimalBinning(
                name=pred_name, dtype="categorical", solver="mip", cat_cutoff=0.05,
                min_prebin_size=0.01, random_state=RANDOM_SEED
            )
    else:
        optb = OptimalBinning(
            name=pred_name, monotonic_trend="auto_asc_desc", 
            dtype="numerical", solver="cp", random_state=RANDOM_SEED,
            min_prebin_size=0.01
            )

    optb.fit(x, y)

    # check the status
    if optb.status != "OPTIMAL":
        logger.warning(f"WoE Binning for {pred_name} has not converged. Status thrown: {optb.status}")

    return optb


def append_binning_table(optb, global_binning_table, pred_name):
    """ 
    Builds binning table based on the fitting of optb and appends it to global_binning_table. 
    Adds a Variable column with pred_name.
    """
    binning_t = optb.binning_table.build()
    binning_t["Attribute"] = pred_name
    if global_binning_table is None:
        global_binning_table = binning_t
    else:
        global_binning_table = pd.concat([global_binning_table, binning_t], axis=0)
    
    return global_binning_table


def check_monotonicity(woe_map, verbose=False) -> bool:
    # Filter binning table
    # drop totals, Special, and Missing
    woe_map = woe_map.loc[
        ~woe_map.Bin.isin(["", "Special", "Missing"])
        ]
    # keep only Numerical (with order)
    woe_map = woe_map.loc[woe_map.Type == "Numerical"]

    # Define a function to check monotonicity
    def check_monotonicity_g(group):
        return (
            group['Event rate'].is_monotonic_increasing
            or group['Event rate'].is_monotonic_decreasing
        )

    # Group by 'group' column and apply the check_monotonic function
    monotonicity_check = woe_map.groupby('Attribute').apply(check_monotonicity_g)

    # If verbose, print out non-monotonic groups
    if verbose:
        non_monotonic_attrs = monotonicity_check[~monotonicity_check].index.tolist()
        if non_monotonic_attrs:
            print(
                f"Non-monotonic event rate for attributes: {non_monotonic_attrs}]"
            )
        else:
            print("All variables display a monotonic event rate.")

    # Return if all groups that display monotonic changes
    return monotonicity_check.all()


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
    cat_binning_table = None
    for cat in cat_columns:
        cat_optb = fit_woe_transform(train, cat, "categorical")
        train[cat] = cat_optb.transform(train[cat], metric="woe")
        test[cat] = cat_optb.transform(test[cat], metric="woe")
        cat_binning_table = append_binning_table(
            cat_optb, cat_binning_table, cat
        )

with Timer("Non parallel WoE for Numerical"):
    num_binning_table = None
    for num in num_columns:
        num_optb = fit_woe_transform(train, num, "numerical")
        train[num] = num_optb.transform(train[num], metric="woe")
        test[num] = num_optb.transform(test[num], metric="woe")
        num_binning_table = append_binning_table(
            num_optb, num_binning_table, num
        )

# Append Woe Mappings
cat_binning_table["Type"] = "Categorical"
num_binning_table["Type"] = "Numerical"
binning_table = pd.concat([cat_binning_table, num_binning_table], axis=0)

# Check monotonicity
monotonicity_check  = check_monotonicity(binning_table, verbose=True)

# Save WoE mapping
binning_table.to_excel(PARENT_DIR / "meta" / "woe_mapping.xlsx", index=False)

# Remove columns with a single WoE value in train
logger.info(
    f"Dropping the variables: {train.columns[train.nunique(axis=0) == 1]} since they result in a single bin after WoE"
    )
train = train.loc[:, train.nunique(axis=0) != 1]
test = test.loc[:, train.columns.values]

# Save processed train/test datasets
train.to_csv(PARENT_DIR / "processed" / "train_apps_woe.csv.zip", index=False)
test.to_csv(PARENT_DIR / "processed" / "test_apps_woe.csv.zip", index=False)
breakpoint()