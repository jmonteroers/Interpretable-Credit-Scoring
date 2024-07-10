import pandas as pd
import numpy as np
import re
from typing import Tuple

from sklearn_get_gini import ExplainableRandomForest
from Estimation.build_scorecard import clean_bins
from utils.attrs import prettify_attrs


def build_scorecard_rf(X, fit_exp_rf: ExplainableRandomForest, binning_table, pdo, peo):
    # Retrieve features from fitted model
    feats = fit_exp_rf.feature_names_in_
    n_feats = len(feats)
    # Build base scorecard
    scorecard = binning_table.loc[binning_table.Attribute.isin(feats)].copy()
    scorecard = scorecard.loc[
        (scorecard["Bin"] != "Special") 
        & ~scorecard["Bin"].isna()
        & (np.abs(scorecard["Count (%)"]) > 1e-8)
    ]
    scorecard.set_index("Attribute", inplace=True)
    # add lead weights columns, filled with zeros by default
    scorecard["Points"] = 0.
    # Compute weights by attribute
    for attribute in scorecard.index.unique():
        print(f"Calculating points for attribute {attribute}...")
        X_attr = scorecard.loc[attribute]
        n_attr = X_attr.shape[0]

        # create artificial observations, 0s everywhere except for attribute
        attr_obs = pd.DataFrame(np.zeros((n_attr, n_feats)), columns=feats)
        attr_obs[attribute] = X_attr.WoE.values

        # edit leaf weights for attribute
        scorecard.loc[attribute, "Points"] = fit_exp_rf.get_feature_score(
            attribute, attr_obs, pdo, peo
            )
    
    # Add Weights - using attribute index to match
    scorecard["Weight (%)"] = fit_exp_rf.get_weight_features(X)
    # TODO: Add WoE
    return scorecard.reset_index()


def extract_elems_num_interval(interval: str) -> Tuple[str, str]:
    # Regular expression pattern to match the interval
    pattern = r'([\[\(].*?), (.*?[\)\]])'
    # Find the match
    match = re.search(pattern, interval)
    if match:
        return match.group(1), match.group(2)
    # Categorical interval
    print(f"No match found for sides of an interval. Interval provided {interval}. Returning emtpy interval")
    return "[", "]"


def extract_elements_cat_interval(interval: str) -> Tuple[str, str]:
    # Use regex to find elements inside the brackets
    match = re.search(r'\[(.*?)\]', interval)
    if match:
        elements = match.group(1).split()
        return elements
    return []


def combine_intervals(scorecard):
    def combine_intv_group(group):
        # detect missing and remove
        has_missing = "Missing" in group.Bin.values
        group_nm = group.loc[group.Bin != "Missing"]
        if len(group_nm) == 0:  # only missing
            merged_intv = "Missing"
        elif group_nm.Type.iloc[0] == "Numerical":
            # extract first interval
            begin_first_intv, _ = extract_elems_num_interval(group_nm.Bin.iloc[0]) 
            # extract last interval
            _, end_last_intv = extract_elems_num_interval(group_nm.Bin.iloc[-1])
            merged_intv = begin_first_intv + ", " + end_last_intv
            merged_intv += has_missing*", Missing"
        else:
            combined_elements = []
            for intv in group_nm.Bin:
                elements = extract_elements_cat_interval(intv)
                combined_elements.extend(elements)
            missing_str = has_missing*" Missing"
            merged_intv = f"[{' '.join(combined_elements)}{missing_str}]"
        return pd.Series({
            "Bin": merged_intv,
            "Count (%)": group["Count (%)"].sum(),
            "Event rate": group["Event"].sum() / group["Count"].sum(),
            "Type": group.Type.iloc[0]
        })
    merged_scorecard = scorecard.groupby(
        ["Attribute", "Weight (%)", "Points"], sort=False
        ).apply(combine_intv_group).reset_index()
    return merged_scorecard


def clean_scorecard_rf(sc):
    """Sorts, cleans and select columns in output order"""
    clean_sc = sc.copy()
    # Sort by Weight, Points
    clean_sc.sort_values(["Weight (%)", "Points"], ascending=[False, True], inplace=True)

    # Edit values for clarity
    clean_sc = prettify_attrs(clean_sc, "Attribute")
    clean_sc.loc[:, "Bad Rate (%)"] = 100*clean_sc["Event rate"]
    clean_sc.loc[:, "Count (%)"] = 100*clean_sc["Count (%)"]
    # Convert bins from list to string
    clean_sc.loc[:, "Bin"] = clean_sc["Bin"].apply(clean_bins)

    # Select columns
    clean_sc = clean_sc[[
        "Attribute", "Weight (%)", "Bin", "Count (%)", "Bad Rate (%)", "Points"
    ]]
    clean_sc.columns = [c.replace(r"%", r"\%") for c in clean_sc.columns]

    return clean_sc


def export_to_latex_rf(sccard, outpath, attributes=None):
    sccard = sccard.copy()
    # filter attributes, if provided
    if attributes is not None:
        sccard = sccard.loc[sccard.Attribute.isin(attributes), :]
    # extra level added to get multirow for the index
    sccard["Index Extra"] = ""
    sccard.set_index(["Attribute", r"Weight (\%)", "Index Extra"], inplace=True)

    # Export to latex
    sccard.style.\
       format(escape="latex").\
       format(subset=[r"Count (\%)", r"Bad Rate (\%)", "Points"], precision=2).\
       format_index(
           {1: lambda f: "{:.2f}".format(f)},
           escape="latex").\
       hide(level=2).\
       to_latex(
           outpath, 
           hrules=True,
           column_format="|p{4cm}p{1.5cm}p{5cm}p{2cm}p{2cm}p{2cm}|",
           multirow_align="t",
           environment="longtable"
           )


if __name__ == "__main__":
    from utils import PARENT_DIR, TARGET
    RANDOM_SEED = 1234

    # Load train data, split into X, y
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")

    X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
    y_train = train[TARGET]

    # Define monotonic constraints - If 0, no constraints
    monotonic_cst = [-1]*X_train.shape[1]

    rf_est_exp = ExplainableRandomForest(
            monotonic_cst=monotonic_cst, 
            max_depth=1, n_estimators=500, min_samples_leaf=0.0025, random_state=RANDOM_SEED)
    rf_est_exp.fit(X_train, y_train)

    # Build scorecard
    bt = pd.read_excel(PARENT_DIR / "meta" / "woe_map" / "woe_mapping.xlsx")
    scorecard = build_scorecard_rf(X_train, rf_est_exp, bt, 20., 600.)
    merged_scorecard = combine_intervals(scorecard)

    # Clean and export scorecard
    clean_sc = clean_scorecard_rf(merged_scorecard)
    clean_sc.to_excel(PARENT_DIR / 'meta' / 'rf_scorecard.xlsx', index=False)
    export_to_latex_rf(clean_sc, PARENT_DIR / 'meta' / 'rf_scorecard_latex.tex', None)
    breakpoint()
