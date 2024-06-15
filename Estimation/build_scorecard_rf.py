import pandas as pd
import numpy as np
from sklearn_get_gini import ExplainableRandomForest
import re
from typing import Tuple


def build_scorecard_rf(fit_exp_rf: ExplainableRandomForest, binning_table, pdo, peo):
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
    # add lead weights columns, filled with zeros by default
    scorecard["LeafWeight"] = 0.
    # Compute weights by attribute
    for attribute in scorecard.Attribute.unique():
        print(f"Calculating points for attribute {attribute}...")
        X_attr = scorecard.loc[scorecard.Attribute == attribute]
        n_attr = X_attr.shape[0]

        # create artificial observations, 0s everywhere except for attribute
        attr_obs = pd.DataFrame(np.zeros((n_attr, n_feats)), columns=feats)
        attr_obs[attribute] = X_attr.WoE.values

        # edit leaf weights for attribute
        scorecard.loc[
            scorecard.Attribute == attribute, 
            "LeafWeight"] = fit_exp_rf.get_feature_score(attribute, attr_obs, pdo, peo)
    return scorecard


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
        has_missing = "Missing" in group.Bin
        group_nm = group.loc[group.Bin != "Missing"]
        if len(group_nm) == 0:  # only missing
            merged_intv = "Missing"
        elif group_nm.Type.iloc[0] == "Numerical":
            # extract first interval
            begin_first_intv, _ = extract_elems_num_interval(group_nm.Bin.iloc[0]) 
            # extract last interval
            _, end_last_intv = extract_elems_num_interval(group_nm.Bin.iloc[-1])
            merged_intv = begin_first_intv + ", " + end_last_intv + has_missing*"+ Missing"
        else:
            combined_elements = []
            for intv in group_nm.Bin:
                elements = extract_elements_cat_interval(intv)
                combined_elements.extend(elements)
            merged_intv = f"[{' '.join(combined_elements)}]"
        return pd.Series({
            "Bin": merged_intv,
            "Count (%)": group["Count (%)"].sum(),
            "Event rate": group["Event"].sum() / group["Count"].sum(),
            "Type": group.Type.iloc[0]
        })
    merged_scorecard = scorecard.groupby(
        ["Attribute", "LeafWeight"], sort=False
        ).apply(combine_intv_group).reset_index()
    # TODO: Add WoE
    # TODO: Adjustments to meet classical style
    return merged_scorecard


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
    scorecard = build_scorecard_rf(rf_est_exp, bt, 20., 600.)
    merged_scorecard = combine_intervals(scorecard)
    breakpoint()
