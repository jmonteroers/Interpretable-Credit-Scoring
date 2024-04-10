import matplotlib.pyplot as plt
from math import sqrt
from statsmodels.formula.api import logit
import numpy as np

from pdb import set_trace


TARGET = "TARGET"


def create_density_by_default(df, feat, bw_method=None, ind=None):
    # separate df into default and non-default
    df_good = df.loc[df[TARGET] == 0]
    df_bad = df.loc[df[TARGET] == 1]
    ax = df_bad[feat].plot.density(bw_method, ind, label="Default")
    df_good[feat].plot.density(bw_method, ind, ax = ax, label="Non-default")
    plt.legend(loc="upper left")
    plt.show()


def create_barplot_perc_def(df, feat):
     """Builds a barplot where the x-axis are the values of `feat`, the y-axis is the % missing"""
     # create copy, to prevent changes to original df
     df = df[[feat, TARGET]].copy()

     # compute mean by default/non-default
     avg_df = df.groupby(feat)[[TARGET]].mean().reset_index()
     avg_df.rename(columns={TARGET: "% Bad"}, inplace=True)

     # compute asymp error, 95% CI
     n = len(df)
     err_df = df.groupby(feat)[[TARGET]].std().reset_index()
     err_df.rename(columns={TARGET: "% Bad"}, inplace=True)
     # z_{0.95} = 1.645
     err_df = 1.645*err_df/sqrt(n)

     # Create barplot
     ax = avg_df.plot.bar(x = feat, y = "% Bad", yerr = err_df, color = "#43ff64d9")
     ax.set_xlabel(feat.capitalize())
     ax.set_ylabel("Proportion Bad")
     # horizontal x-axis labels
     plt.xticks(rotation=0)

     plt.show()


def get_ftest(df, feat):
    logit_model = logit(f"{TARGET} ~ {feat}", data = df)
    try:
        logit_fit = logit_model.fit()
    except Exception as e:
        print(f"Error fitting logit for {feat}, returning p-value of 1.")
        return 1.
    # carry out f-test
    p = len(logit_fit.params)
    contrast_mx = np.identity(p)[1:]
    return logit_fit.f_test(contrast_mx).pvalue



def export_significance_test(df, outpath):
    """Writes the F-test of an overall significance test of a simple regression on TARGET on a predictor"""
    # Define List of Variables to Include
    excl_vars = ["SK_ID_CURR", "TARGET", "Unnamed: 0"]
    report_vars = [col for col in df.columns if col not in excl_vars]

    # Compute overall significance
    outdf = pd.DataFrame({"Variable": report_vars})

    outdf["P-Values"] = outdf["Variable"].apply(lambda v: get_ftest(df, v))

    # Sort by p-value, filter?
    outdf.sort_values("P-Values", inplace=True)

    # Save to output file
    outdf.style.\
        format(escape="latex").\
        format(subset="P-Values", precision=3).\
        hide(axis="index").\
        to_latex(outpath, hrules=True)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from pathlib import Path

    PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
    df = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_ext.csv.zip')

    # F-test table with p-values
    export_significance_test(df, PARENT_DIR / "meta" / "bivariate_significance_pvals.tex")
    # plots to illustrate bivariate effects
    create_density_by_default(df, "LTV", bw_method="silverman", ind=np.arange(.1, 2., .01))
    create_barplot_perc_def(df, "FLAG_WORK_PHONE")
    set_trace()

