import pandas as pd
import numpy as np
from math import log
from sklearn.linear_model import LogisticRegression
import re
from itertools import product
import matplotlib.pyplot as plt

from utils import TARGET, attrs
from utils.pd import concat_train_test
from utils.stats import ecdf

pd.options.mode.copy_on_write = True

def build_scorecard(df, binning_table, estimator=None, pdo=20., peo=600.):
    """
    Input:
     - df is the applications table
     - binning_table is the table with Bin, WoE, and Attribute
     - estimator is used to obtain the coefficients of the model

    Return:
    binning_table extended with the computed Points and Coefficients
    """
    # Step 1: Fit logistic model
    X = df.drop(TARGET, axis=1)
    # 1s are good loans
    y = pd.to_numeric(df.TARGET)
    estimator = LogisticRegression() if estimator is None else estimator
    estimator.fit(X, y)

    # Step 2: Build coefficients + weights dataframe
    coefficients = estimator.coef_[0]
    n_vars = len(coefficients)
    intercept = estimator.intercept_[0]
    coefficients_df = pd.DataFrame({
        'Attribute': X.columns, 
        'Coefficient': coefficients,
        'Weight (%)': get_weights(X, coefficients)
        })

    # Step 3: Add coefficients to binning table
    bt_ext = binning_table.copy()
    bt_ext = pd.merge(
        binning_table, coefficients_df, how='left', on='Attribute'
        )

    # Step 4: Compute points
    factor = -pdo/log(2)
    bt_ext["Points"] = (
        bt_ext["Coefficient"]*bt_ext["WoE"]*factor
        + (
            intercept*factor + peo
        ) / n_vars
    )

    return bt_ext, intercept


def get_weights(X, coeffs):
    std_X = np.std(X, axis=0)
    wt_X = std_X * coeffs
    # Normalise (%)
    return 100 * wt_X / wt_X.sum()


def check_scorecard(bt_with_points):
    """
    Prints out the attributes without points, if any. Also prints out how many attributes have points
    """
    # remove Totals for Check
    bins_without_points = bt_with_points.loc[
        ~bt_with_points["Bin"].isna() & bt_with_points["Points"].isna()
    ]
    attrs_without_points = bins_without_points.Attribute.unique()
    if len(attrs_without_points):
        print("The following attributes do not have points in the scorecard:")
        for attr in attrs_without_points:
            print(f"- {attr}")
    
    # how many attributes have points
    n_attrs_points = bt_with_points.Attribute.nunique() - len(attrs_without_points)
    print(f"Points have been successfully added to {n_attrs_points} attributes.")


def clean_bins(str_l, max_len=1):
        """max_len does not include the extra ... added"""
        if not ("[" in str_l and "]" in str_l):
            return str_l
        # Parse string to list
        str_l = str_l.strip("[]")
        if r"'" in str_l:
            pattern = r"'(.*?)'"
        # assume numbers
        else:
            pattern = r"(\d+)"
        l = re.findall(pattern, str_l)
        # Correct if too long
        has_missing = "Missing" in l  # never drop missing
        if len(l) >= max_len:
            l = l[:max_len] + has_missing*["Missing"] + ["..."]
        return ", ".join(l)


def clean_scorecard(bt_with_points):
    """Remove special, bins with zero count. Cleans and select columns in output order"""
    clean_sc = bt_with_points.loc[
        (bt_with_points["Bin"] != "Special") 
        & ~bt_with_points["Points"].isna()
        & (np.abs(bt_with_points["Count (%)"]) > 1e-8)
    ]

     # Sort by Weight(%), Points
    clean_sc.sort_values(["Weight (%)", "Points"], ascending=[False, True], inplace=True)

    # Edit values for clarity
    clean_sc = attrs.prettify_attrs(clean_sc, "Attribute")
    clean_sc.loc[:, "Bad Rate (%)"] = 100*clean_sc["Event rate"]
    clean_sc.loc[:, "Count (%)"] = 100*clean_sc["Count (%)"]
    # Convert bins from list to string
    clean_sc.loc[:, "Bin"] = clean_sc["Bin"].apply(clean_bins)

    # Select columns
    clean_sc = clean_sc[[
        "Attribute", "Coefficient" , "Weight (%)", "Bin", "Count (%)", "Bad Rate (%)", "WoE", "Points"
    ]]
    clean_sc.columns = [c.replace(r"%", r"\%") for c in clean_sc.columns]

    return clean_sc


def export_to_latex(sccard, outpath, attributes=None):
    # filter attributes, if provided
    if attributes is not None:
        sccard = sccard.loc[sccard.Attribute.isin(attributes), :]
    # extra level added to get multirow for all three levels of the index
    sccard["Index Extra"] = ""
    sccard.set_index(["Attribute", "Coefficient" , r"Weight (\%)", "Index Extra"], inplace=True)

    # Export to latex
    sccard.style.\
       format(escape="latex").\
       format(subset=[r"Count (\%)", r"Bad Rate (\%)", "WoE", "Points"], precision=2).\
       format_index({
           1: lambda f: "{:.2f}".format(f),
           2: lambda f: "{:.2f}".format(f),
           },
           escape="latex").\
       hide(level=3).\
       to_latex(
           outpath, 
           hrules=True,
           column_format="|p{4cm}p{1.25cm}p{1.25cm}p{3cm}p{1.25cm}p{1.25cm}p{1.1cm}p{1.1cm}|",
           multirow_align="t",
           environment="longtable"
           )


def score_woe_applications(X, scorecard, intercept, pdo=20., peo=600.):
    # Summarise scorecard by coefficients
    scorecard = scorecard.loc[~scorecard["Points"].isna()]  # remove non-scored attributes
    coeffs = scorecard.groupby("Attribute")[["Coefficient"]].mean().reset_index()
    # Reorder X to have the Attributes in the scorecard
    X = X[coeffs.Attribute.values.tolist()]
    # Get vector of coefficients
    coeffs = coeffs.Coefficient.values.reshape(-1, 1)
    # Obtain score
    factor = -pdo/log(2)
    return factor * (X.values @ coeffs + intercept) + peo


def dens_plot_comparison(train_sc, test_sc):
    """
    Create density plot by default/non-default, train/test dataset. 
    `train_sc`and `test_sc` must contain Score attribute.
    """
    # Concatenate train and test datasets, using intersection of columns
    df_scores = concat_train_test(train_sc, test_sc)

    # Iterate over Default x Train/Test, creating a histogram for each
    df_scores["Default"] = df_scores[TARGET].map({0: "Non-default", 1: "Default"})
    options = product(["Train", "Test"], ["Default", "Non-default"])
    ax = None
    for option in options:
        df_sel = df_scores.loc[
            (df_scores.Dataset == option[0]) & (df_scores.Default == option[1])
            ]
        label_sel = f"{option[0]}, {option[1]}"
        ax = df_sel["Score"].plot.density(
            bw_method=0.5,
            label=label_sel,
            ax = ax
            )
    ax.legend()
    ax.set_xlabel("Score")
    plt.show()


def ecdf_plot_comparison(train_sc, test_sc):
    # Set up figure
    plt.figure(figsize=(8, 6))
    plt.xlabel('Score')
    plt.ylabel('ECDF')
    plt.title('Empirical Cumulative Distribution Function (ECDF)')

    # Concatenate train and test datasets, using intersection of columns
    df_scores = concat_train_test(train_sc, test_sc)

    df_scores["Default"] = df_scores[TARGET].map({0: "Non-default", 1: "Default"})
    options = product(["Train", "Test"], ["Default", "Non-default"])
    for option in options:
        df_sel = df_scores.loc[
            (df_scores.Dataset == option[0]) & (df_scores.Default == option[1])
            ]
        label_sel = f"{option[0]}, {option[1]}"
        x, y = ecdf(df_sel["Score"])
        plt.plot(x, y, label=label_sel)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    from utils.utils import PARENT_DIR

    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_bic_npos.csv.gz', compression="gzip")
    bt = pd.read_excel(PARENT_DIR / "meta" / "woe_map" / "woe_mapping.xlsx")

    scorecard, _ = build_scorecard(train, bt)
    check_scorecard(scorecard)
    scorecard = clean_scorecard(scorecard)

    # Export scorecard as Excel
    scorecard.to_excel(PARENT_DIR / 'meta' / 'scorecard_bic.xlsx', index=False)
    export_to_latex(scorecard, PARENT_DIR / 'meta' / 'scorecard_latex.tex', None)

    breakpoint()
    # Score train applications
    # Must use raw scorecard so that attributes in scorecard and scored datasets match
    raw_scorecard, intercept = build_scorecard(train, bt)
    train["Score"] = score_woe_applications(train, raw_scorecard, intercept)

    # Score test applications
    test = pd.read_csv(PARENT_DIR / "processed" / "test_apps_woe.csv.zip", compression="zip")
    test["Score"] = score_woe_applications(test, raw_scorecard, intercept)

    ecdf_plot_comparison(train, test)
    breakpoint()

    dens_plot_comparison(train, test)
    breakpoint()

