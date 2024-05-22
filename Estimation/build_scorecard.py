import pandas as pd
from math import log
from sklearn.linear_model import LogisticRegression

TARGET = "TARGET"

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
    y = pd.to_numeric(df.TARGET == 0)
    estimator = LogisticRegression() if estimator is None else estimator
    estimator.fit(X, y)

    # Step 2: Build coefficients dataframe
    coefficients = estimator.coef_[0]
    n_vars = len(coefficients)
    intercept = estimator.intercept_[0]
    coefficients_df = pd.DataFrame({'Attribute': X.columns, 'Coefficient': coefficients})

    # Step 3: Add coefficients to binning table
    bt_ext = binning_table.copy()
    bt_ext = pd.merge(
        binning_table, coefficients_df, how='left', on='Attribute'
        )

    # Step 4: Compute points
    bt_ext["Points"] = (
        bt_ext["Coefficient"]*bt_ext["WoE"]*pdo/log(2)
        + (
            intercept*pdo/log(2) + peo
        ) / n_vars
    )

    return bt_ext


def add_weights(bt_ext):
    # Step 1: summarise by attribute, find average points per attribute
    weighted_avg = bt_ext.groupby('Attribute').apply(
        lambda x: (x['Points'] * x['Count (%)']).sum() / x['Count (%)'].sum()
        )
    weighted_avg.rename("Average Points", inplace=True)
    weighted_avg = weighted_avg.reset_index()

    # Step 2: Normalise the average points per attribute, getting weights (%)
    weighted_avg["Weight (%)"] = 100*(
        weighted_avg["Average Points"] 
        / weighted_avg["Average Points"].sum()
    )

    # Step 3: Merge back with scorecard
    bt_weights = bt_ext.copy()
    bt_weights = pd.merge(bt_weights, weighted_avg, how="left", on="Attribute")
    
    return bt_weights


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


def clean_scorecard(bt_with_points):
    """Remove 'Special' bin and rows without points, cleans and select columns in output order"""
    clean_sc = bt_with_points.loc[
        (bt_with_points["Bin"] != "Special") & ~bt_with_points["Points"].isna()
    ]
    # Edit values for clarity
    clean_sc.loc[:, "Attribute"] = clean_sc["Attribute"].str.capitalize()
    clean_sc.loc[:, "Bad Rate (%)"] = 100*clean_sc["Event rate"]
    clean_sc.loc[:, "Count (%)"] = 100*clean_sc["Count (%)"]
    # Select columns
    clean_sc = clean_sc[[
        "Attribute", "Bin", "Count (%)", "Bad Rate (%)", "Coefficient", "WoE", "Points", "Weight (%)"
    ]]
    return clean_sc


def export_to_latex(sccard, outpath, attributes=None):
    # filter attributes, if provided
    if attributes is not None:
        sccard = sccard.loc[sccard.Attribute.isin(attributes), :]
    # Export to latex
    sccard.style.\
       format(escape="latex").\
       format(subset=["Count (%)", "Bad Rate (%)", "Coefficient", "WoE", "Points", "Weight (%)"], precision=2).\
       hide(axis="index").\
       to_latex(outpath, hrules=True)


if __name__ == "__main__":
    from pathlib import Path

    PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_bic.csv.gz', compression="gzip")
    bt = pd.read_excel(PARENT_DIR / "meta" / "woe_mapping.xlsx")

    scorecard = build_scorecard(train, bt)
    check_scorecard(scorecard)
    scorecard = add_weights(scorecard)
    scorecard = clean_scorecard(scorecard)

    # Export scorecard as Excel
    scorecard.to_excel(PARENT_DIR / 'meta' / 'scorecard_bic.xlsx', index=False)
    export_to_latex(scorecard, PARENT_DIR / 'meta' / 'scorecard_latex.tex', ["Ext_source_3", "Name_education_type"])
    breakpoint()

