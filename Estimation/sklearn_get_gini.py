from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.special import logit, expit
import numpy as np
import pandas as pd
from collections import defaultdict


class ExplainableRandomForest(RandomForestClassifier):
    """Inherit from Random Forest Classifier, modify predict_proba method to be based on average logit prediction"""
    def predict_using_logit(self, X, return_prob=True, estimators=None):
        # Check data
        X = self._validate_X_predict(X)
        if estimators is None:
            estimators = self.estimators_
        n_est = len(estimators)
        # avoid storing the output of every estimator by summing them here
        logit_estimates = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64) 
        for tree in estimators:
            logit_estimates += logit(tree.predict_proba(X))
        # Average logit estimates
        logit_estimates = logit_estimates/n_est
        # Transform back to probability if required
        estimates = expit(logit_estimates) if return_prob else logit_estimates
        return estimates
    
    def compute_trees_by_feature(self, reload=False):
        if not reload and hasattr(self, "trees_by_feature"):
            return
        self.trees_by_feature = defaultdict(lambda: [])
        # Here assuming decision stumps have been fitted
        for tree in self.estimators_:
            idx_feat = tree.tree_.feature[0]
            feat = self.feature_names_in_[idx_feat]
            self.trees_by_feature[feat].append(tree)
    
    def get_feature_logit(self, feature, X):
        if not hasattr(self, "trees_by_feature"):
            self.compute_trees_by_feature()
        trees_feat = self.trees_by_feature.get(feature)
        if trees_feat is None:
            return pd.NA
        return self.predict_using_logit(X, estimators=trees_feat, return_prob=False)[:, 1]
    
    def get_feature_score(self, feature, X, pdo=20., peo=600.):
        K = X.shape[1]
        factor = -pdo/np.log(2)
        return factor*self.get_feature_logit(feature, X) + peo/K
    
    def get_weight_features(self, X):
        """Compute % share of sum of sd of logit scores for each feature as weights. 
        Note that the result does not depend on pdo and peo by the properties of the standard deviation"""
        if not hasattr(self, "trees_by_feature"):
            self.compute_trees_by_feature()
        sd_preds = pd.Series()
        for feature in self.trees_by_feature.keys():
            feat_scores = self.get_feature_score(feature, X)
            sd_preds[feature] = np.std(feat_scores)
        # Normalise weights and return
        return 100. * sd_preds / sd_preds.sum()




def gini_train_test(fit, X_train, y_train, X_test, y_test, pred_proba=None):
    pred_proba = fit.predict_proba if pred_proba is None else pred_proba
    # Get Gini, train
    train_probs = pred_proba(X_train)[:, 1]
    train_gini = 2*roc_auc_score(y_train, train_probs) - 1.

    # Get Gini, test
    test_probs = pred_proba(X_test)[:, 1]
    test_gini = 2*roc_auc_score(y_test, test_probs) - 1.

    return train_gini, test_gini


def get_gini_sklearn(estimator, X_train, y_train, X_test, y_test, param_grid, cv=5, verbose=0):
    # Estimate by CV Grid Search using train
    est_grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", verbose=verbose)
    est_grid_search.fit(X_train, y_train)
    est_best = est_grid_search.best_estimator_

    # Get Gini, train, test
    train_gini, test_gini = gini_train_test(est_best, X_train, y_train, X_test, y_test)

    return est_best, train_gini, test_gini


if __name__ == "__main__":
    from utils import PARENT_DIR, TARGET

    RANDOM_SEED = 1234

    # Which models to get Gini for
    FIT_DT = False
    FIT_RF = False
    FIT_BOOST = False
    FIT_NN = True
    MONOTONICITY = True

    # Store gini metrics
    gini_res = {}

    # Load data, split into X, y
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
    test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")

    X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET, "SK_ID_CURR"])
    y_test = test[TARGET]

    # Define monotonic constraints - If 0, no constraints
    monotonic_val = -1 if MONOTONICITY else 0
    monotonic_cst = [monotonic_val]*X_train.shape[1]

    #### Decision Tree ####
    if FIT_DT:
        from sklearn.tree import DecisionTreeClassifier
        dt_param_grid = {
        'max_depth': [3, 10, 15],
        'min_samples_leaf': [0.0025, 0.005, 0.01]
        }
        dt_est = DecisionTreeClassifier(monotonic_cst=monotonic_cst, random_state=RANDOM_SEED)
        dt_fit, dt_train_gini, dt_test_gini = get_gini_sklearn(dt_est, X_train, y_train, X_test, y_test, dt_param_grid, cv=5, verbose=4
            )
        print(f"Decision Tree. train_gini: {dt_train_gini}; test_gini: {dt_test_gini}")
        gini_res["DT"] = (dt_train_gini, dt_test_gini)

        ## Explainable - reduce max depth
        dt_est_exp = DecisionTreeClassifier(
            monotonic_cst=monotonic_cst, 
            max_depth=5, min_samples_leaf=0.005, random_state=RANDOM_SEED
            )
        dt_est_exp.fit(X_train, y_train)
        dt_exp_train_gini, dt_exp_test_gini = gini_train_test(dt_est_exp, X_train, y_train, X_test, y_test)
        print(f"Explainable Decision Tree. train_gini: {dt_exp_train_gini}; test_gini: {dt_exp_test_gini}")
        gini_res["Exp_DT"] = (dt_exp_train_gini, dt_exp_test_gini)

        breakpoint()

    #### Random Forest ####
    if FIT_RF:
        from sklearn.ensemble import RandomForestClassifier
        rf_param_grid = {
        'n_estimators': [10, 100, 1000],
        'max_depth': [1, 5, 10],
        'min_samples_leaf': [0.0025, 0.005, 0.01]
        }
        # Simple grid (testing)
        # rf_param_grid = {
        #     "n_estimators": [100],
        #     "max_depth": [1],
        #     "min_samples_leaf": [0.005]
        # }
        rf_est = RandomForestClassifier(monotonic_cst=monotonic_cst, random_state=RANDOM_SEED)
        rf_fit, rf_train_gini, rf_test_gini = get_gini_sklearn(
            rf_est, X_train, y_train, X_test, y_test, rf_param_grid, cv=3, verbose=4
            )
        print(f"Random Forest. train_gini: {rf_train_gini}; test_gini: {rf_test_gini}")
        gini_res["RF"] = (rf_train_gini, rf_test_gini)
        breakpoint()
        ## Explainable - set max_depth to 1, n_estimators to 100
        rf_est_exp = ExplainableRandomForest(
            monotonic_cst=monotonic_cst, 
            max_depth=1, n_estimators=100, min_samples_leaf=0.005, random_state=RANDOM_SEED)
        rf_est_exp.fit(X_train, y_train)
        rf_exp_train_gini, rf_exp_test_gini = gini_train_test(
            rf_est_exp, X_train, y_train, X_test, y_test, pred_proba=rf_est_exp.predict_using_logit
            )
        print(f"Explainable Random Forest. train_gini: {rf_exp_train_gini}; test_gini: {rf_exp_test_gini}")
        gini_res["Exp_RF"] = (rf_exp_train_gini, rf_exp_test_gini)

        breakpoint()


    #### Gradient Boosting ####
    if FIT_BOOST:
        from xgboost import XGBClassifier

        # Monotone Constraints
        gb_constraints = None
        if MONOTONICITY:
            gb_constraints = {feat: -1 for feat in X_train.columns}
        
        gb_param_grid = {
        'max_depth': [1, 3, 5],
        'n_estimators': [75, 100, 250],
        'learning_rate': [0.1, 0.3, 0.5]
        }
        # Simple grid (testing)
        gb_param_grid = {
        'max_depth': [1],
        'n_estimators': [100],
        'learning_rate': [0.1]
        }
        gb_est = XGBClassifier(tree_method="hist", monotone_constraints=gb_constraints, random_state=RANDOM_SEED)
        gb_fit, gb_train_gini, gb_test_gini = get_gini_sklearn(
            gb_est, X_train, y_train, X_test, y_test, gb_param_grid, cv=3, verbose=4
            )
        print(f"Gradient Boosting. train_gini: {gb_train_gini}; test_gini: {gb_test_gini}")
        gini_res["GB"] = (gb_train_gini, gb_test_gini)
        breakpoint()
        ## Explainable - set max_depth to 1, n_estimators to 100
        gb_est_exp = XGBClassifier(
            tree_method="hist",
            monotone_constraints=gb_constraints, 
            max_depth=1, n_estimators=100, learning_rate=0.5, 
            random_state=RANDOM_SEED
            )
        gb_est_exp.fit(X_train, y_train)
        gb_exp_train_gini, gb_exp_test_gini = gini_train_test(
            gb_est_exp, X_train, y_train, X_test, y_test
            )
        print(f"Explainable Gradient Boosting. train_gini: {gb_exp_train_gini}; test_gini: {gb_exp_test_gini}")
        gini_res["Exp_GB"] = (gb_exp_train_gini, gb_exp_test_gini)

        breakpoint()
    
    if FIT_NN:
        from sklearn.neural_network import MLPClassifier
        nn_param_grid = {
        'learning_rate': ['invscaling', 'adaptive'],
        'learning_rate_init': [0.01, 0.001, 0.0001]
        }
        nn_est = MLPClassifier(max_iter=100, early_stopping=True, random_state=RANDOM_SEED)
        nn_fit, nn_train_gini, nn_test_gini = get_gini_sklearn(
            nn_est, X_train, y_train, X_test, y_test, nn_param_grid, cv=5, verbose=4
            )
        print(f"Neural Network. train_gini: {nn_train_gini}; test_gini: {nn_test_gini}")
        gini_res["NN"] = (nn_train_gini, nn_test_gini)
        breakpoint()

    
    ## Compare standard prediction against new prediction method, Random Forest
    from utils.plot import scatterplot_compare_series
    rf_est_exp = ExplainableRandomForest(
            monotonic_cst=monotonic_cst, 
            max_depth=1, n_estimators=100, min_samples_leaf=0.005, random_state=RANDOM_SEED)
    rf_est_exp.fit(X_train, y_train)

    std_probs = rf_est_exp.predict_proba(X_train)[:, 1]
    new_probs = rf_est_exp.predict_using_logit(X_train)[:, 1]
    breakpoint()
    scatterplot_compare_series(
        std_probs, new_probs, 
        x_label="Scikit-learn Probabilities", y_label="Probabilities using logit", 
        s=20)
    



