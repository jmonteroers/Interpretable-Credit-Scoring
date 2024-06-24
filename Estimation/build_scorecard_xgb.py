import numpy as np
import xgboost as xgb
import pandas as pd
from collections import defaultdict
from typing import Iterable


def get_trees_by_feature(booster) -> dict:
    """Extracts trees from booster, building a dictionary that maps a feature name to a list of trees. 
    Assumes depth of fitted trees is 1"""
    out_dict = defaultdict(lambda: [])
    for tree in booster:
        # Get splitting feature
        feat = tree.trees_to_dataframe().loc[0, "Feature"] 
        # Add to mapping
        out_dict[feat] += [tree]
    return out_dict


def get_points_for_feature(
        trees_by_feat: dict, feat: str, X: pd.DataFrame, 
        val: Iterable, base_score: float, pdo: float, peo: float):
    """Calculate the points for a vector of feature values"""
    n = X.shape[0]
    K = len(trees_by_feat)
    factor = -pdo/np.log(2)
    trees = trees_by_feat[feat]
    # Build artificial observation
    X_val = X.copy()
    X_val[feat] = val
    # Get estimated points for this feature value
    val_points = (base_score + peo) / K
    # base_margin set to zero so that we extract margin values
    DM_X_val = xgb.DMatrix(X_val, base_margin=np.zeros((n,)))
    for tree in trees:
        val_points += factor*tree.predict(DM_X_val, output_margin=True)
    return val_points


def build_scorecard_xgb(
        binning_table: pd.DataFrame, trees_by_feat: dict, 
        feats: list, base_score: float, pdo: float, peo: float
        ) -> pd.DataFrame:
    # Retrieve features from fitted model
    tree_feats = list(trees_by_feat.keys())
    # Build base scorecard
    scorecard = binning_table.loc[binning_table.Attribute.isin(tree_feats)].copy()
    scorecard = scorecard.loc[
        (scorecard["Bin"] != "Special") 
        & ~scorecard["Bin"].isna()
        & (np.abs(scorecard["Count (%)"]) > 1e-8)
    ]
    scorecard.set_index("Attribute", inplace=True)
    # add points column, filled with zeros by default
    scorecard["Points"] = 0.
    # Compute weights by attribute
    for attribute in scorecard.index.unique():
        print(f"Calculating points for attribute {attribute}...")
        sc_attr = scorecard.loc[attribute]
        n_rows_sc = sc_attr.shape[0]

        # create artificial X, 0s everywhere except for attribute
        attr_obs = pd.DataFrame(np.zeros((n_rows_sc, len(feats))), columns=feats)

        # edit points for attribute
        scorecard.loc[attribute, "Points"] = get_points_for_feature(
            trees_by_feat, attribute, attr_obs, sc_attr.WoE.values,
            base_score=base_score, pdo=pdo, peo=peo
        )
    return scorecard.reset_index()


def calc_weights_point_based(X: pd.DataFrame, trees_by_feat: dict, base_score: float) -> pd.Series:
    "Note that the result does not depend on pdo, peo, that are simply given some values"
    sd_by_feat = pd.Series()
    for feat in trees_by_feat:
        points = get_points_for_feature(
            trees_by_feat, feat, X, X[feat], base_score, pdo=20., peo=600.
            )
        sd_by_feat[feat] = np.std(points)
    return sd_by_feat / sd_by_feat.sum()


if __name__ == "__main__":
    from xgboost import XGBClassifier
    from utils import PARENT_DIR, TARGET
    from build_scorecard_rf import combine_intervals, clean_scorecard_rf, export_to_latex_rf
    RANDOM_SEED = 1234

    # Load train data
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
    X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
    y_train = train[TARGET]
    
    gb_constraints = {feat: -1 for feat in X_train.columns}

    gb_est_exp = XGBClassifier(
            tree_method="hist",
            monotone_constraints=gb_constraints, 
            max_depth=1, n_estimators=100, learning_rate=0.5, 
            random_state=RANDOM_SEED,
            base_score=0.5
            )
    gb_est_exp.fit(X_train, y_train)

    xgb_trees = get_trees_by_feature(gb_est_exp.get_booster())
    # function to iterate over woe mapping and calculate points 
    ltv_points = get_points_for_feature(xgb_trees, "LTV", X_train.iloc[0:3], [-0.52, 0.11, 0.25], 0., 20., 600.)

    # Build scorecard
    bt = pd.read_excel(PARENT_DIR / "meta" / "woe_map" / "woe_mapping.xlsx")
    feats = X_train.columns.tolist()
    scorecard = build_scorecard_xgb(bt, xgb_trees, feats, 0., 20., 600.)
    # Add Weight (%)
    scorecard.set_index("Attribute", inplace=True)
    scorecard["Weight (%)"] = calc_weights_point_based(X_train, xgb_trees, 0.)
    # TODO: Add WoE - maybe separate function
    scorecard.reset_index(inplace=True)

    # Clean-up
    merged_scorecard = combine_intervals(scorecard)
    clean_sc = clean_scorecard_rf(merged_scorecard)

    # Export
    clean_sc.to_excel(PARENT_DIR / 'meta' / 'xgb_scorecard.xlsx', index=False)
    export_to_latex_rf(clean_sc, PARENT_DIR / 'meta' / 'xgb_scorecard_latex.tex', None)

    breakpoint()