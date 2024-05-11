import pandas as pd
from math import log
from sklearn.linear_model import LogisticRegression

TARGET = "TARGET"

def build_scorecard(df, binning_table, estimator=None, pdo=20., peo=600.):
    """
    Input:
     - df is the applications table
     - binning_table is the table with Bin, WoE, and Attribute
     - estimator is used to obtain the coefficients of the model

    Return:
    binning_table extended with the computed Points
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


if __name__ == "__main__":
    from pathlib import Path

    PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
    train = train.drop(columns=["SK_ID_CURR"])
    bt = pd.read_excel(PARENT_DIR / "meta" / "woe_mapping.xlsx")

    scorecard = build_scorecard(train, bt)
    check_scorecard(scorecard)
    breakpoint()

