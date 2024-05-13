"""
The purpose of this script is to provide a function to analyse the effect of 
a continuous variable on default by quantiles (discretising the continuous feature by quantiles)

In particular, the following variables are being analysed due to zero-IV:
['N_BUREAU_CURR_BAD_30', 'N_BUREAU_CURR_BAD_60', 'N_HC_BAD_30_CURR',
       'N_HC_BAD_30_QRT', 'N_HC_BAD_30_YR', 'COUNT_CC_OVER_LIMIT_CURR',
       'TIMES_CC_OVER_LIMIT_QRT']
"""
import pandas as pd

from bivariate_report_significance import create_barplot_perc_def

TARGET = "TARGET"


def plot_barplot_by_quantiles(df, feat, n_quantiles=4):
    df = df.loc[:, [feat, TARGET]].copy()
    # Discretise by quantiles
    df[feat] = pd.qcut(df[feat], q=n_quantiles, duplicates="drop")
    create_barplot_perc_def(df, feat)


if __name__ == "__main__":
    from pathlib import Path
    
    # Load input data for WoE
    PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
    df = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_imp.csv.gz', compression="gzip")

    # Create barplots by quantiles
    plot_barplot_by_quantiles(df, "N_BUREAU_CURR_BAD_30", n_quantiles=100)
    plot_barplot_by_quantiles(df, "N_HC_BAD_30_YR", n_quantiles=100)
    # Maybe for times cc over limite is worth exploring thinner bins!
    plot_barplot_by_quantiles(df, "TIMES_CC_OVER_LIMIT_QRT", n_quantiles=100)

    # Print frequency table
    # Cannot have more than one bin due to high frequency of zero value
    # requires about 200 quantiles (0.5% min size bins)
    print(df["N_BUREAU_CURR_BAD_30"].value_counts())
    # requires about 1000 quantiles (0.1% min size bins)
    print(df["N_HC_BAD_30_YR"].value_counts())
    # with minimum 1% min size it displays clear effect on defaults
    print(df["TIMES_CC_OVER_LIMIT_QRT"].value_counts())
    