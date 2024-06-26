from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from utils.utils import PARENT_DIR, TARGET

RANDOM_SEED = 1234
FIT_NN = True
FIT_XGB = True
FIT_EXP_XGB = True

# Load data, split into X, y
train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")
    
X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
y_train = train[TARGET]

# Prepare sample for plot
sample_X_train = X_train.sample(n=75, random_state=RANDOM_SEED)

def plot_ice(model, X, features, x_labels):
    axes = PartialDependenceDisplay.from_estimator(
        model, X, features=features, centered=True, kind='both', pd_line_kw={"label": "Average"}
        ).axes_.squeeze()
    for ax, xlab in zip(axes, x_labels):
        # Add horizontal line
        ax.axhline(y=0, color='red', linestyle='--', label="Non-increasing Limit", alpha=0.75)
        ax.legend()
        # Modify x label
        ax.set_xlabel(xlab)
    return ax

# Neural Network Model
if FIT_NN:
    nn_model = MLPClassifier(
        learning_rate="invscaling", learning_rate_init=0.001, 
        max_iter=100, early_stopping=True, random_state=RANDOM_SEED
        )
    nn_model.fit(X_train, y_train)
    plot_ice(
        nn_model, sample_X_train, features=["LTV", "NAME_EDUCATION_TYPE"], 
        x_labels=["Loan-to-Value", "Education Level"]
        )
    plt.show()

# General Gradient Boosting Model
if FIT_XGB:
    gb_constraints = {feat: -1 for feat in X_train.columns}
    xgb_model = XGBClassifier(
            tree_method="hist",
            monotone_constraints=gb_constraints, 
            max_depth=5, n_estimators=250, learning_rate=0.1, 
            random_state=RANDOM_SEED
            )
    xgb_model.fit(X_train, y_train)
    plot_ice(
            xgb_model, sample_X_train, features=["LTV", "NAME_EDUCATION_TYPE"], 
            x_labels=["Loan-to-Value", "Education Level"]
            )
    plt.show()

# Explainable Gradient Boosting Model
if FIT_EXP_XGB:
    gb_constraints = {feat: -1 for feat in X_train.columns}
    exp_xgb_model = XGBClassifier(
            tree_method="hist",
            monotone_constraints=gb_constraints, 
            max_depth=1, n_estimators=100, learning_rate=0.3, 
            random_state=RANDOM_SEED
            )
    exp_xgb_model.fit(X_train, y_train)
    plot_ice(
        exp_xgb_model, sample_X_train, features=["LTV", "NAME_EDUCATION_TYPE"], 
        x_labels=["Loan-to-Value", "Education Level"]
        )    
    plt.show()

breakpoint()