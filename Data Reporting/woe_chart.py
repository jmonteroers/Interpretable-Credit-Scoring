import pandas as pd
from pathlib import Path
import logging
from optbinning import OptimalBinning
import matplotlib.pyplot as plt

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
logger = logging.getLogger("WOE_logger")
logging.basicConfig(level=logging.INFO)

TARGET = "TARGET"
RANDOM_SEED = 1234


df = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_imp.csv.gz', compression="gzip")

# Fitting
x = df["LTV"].values
y = df[TARGET].values
optb = OptimalBinning(
    name="LTV", dtype="numerical", solver="cp", 
    monotonic_trend="auto_asc_desc",
    min_prebin_size=0.01,
    random_state=RANDOM_SEED
    )
optb.fit(x, y)
# check the status
if optb.status != "OPTIMAL":
    logger.warning(f"WoE Binning for LTV has not converged. Status thrown: {optb.status}")

# Create chart
binning_table = optb.binning_table
binning_table.build()
binning_table.plot(metric="woe", show_bin_labels=True)
plt.show()

# Comparison vs existing WoE - same!
woe_mapping = pd.read_excel(PARENT_DIR / 'meta' / 'woe_mapping.xlsx')
woe_mapping = woe_mapping.loc[woe_mapping["Attribute"] == "LTV", ]
print(woe_mapping)

