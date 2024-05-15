import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from pathlib import Path

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
TARGET = "TARGET"

# Load Data
train_aic = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_aic.csv.gz', compression="gzip")
train_bic = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_bic.csv.gz', compression="gzip")
test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")

# Replace targets
train_aic[TARGET] = train_aic[TARGET] == 0.
train_bic[TARGET] = train_bic[TARGET] == 0.
test[TARGET] = test[TARGET] == 0.

def fit_logit(train):
    # Add constants
    train = sm.add_constant(train)
    # Get Xs
    X_train = train.drop(columns=TARGET)
    # Estimate Logistic Regressions
    logit = sm.Logit(train.TARGET == 0., X_train)
    return logit.fit()


def get_roc_auc(logit_fit, test):
    # Get X_test
    pred_names = logit_fit.params.index.tolist()[1:]  # removing intercept col
    X_test = test.loc[:, pred_names]
    # Add constant to X_test
    X_test = sm.add_constant(X_test)
    # Make predictions, obtain ROC AUC
    y_score = logit_fit.predict(X_test)
    return roc_auc_score(test.TARGET.values == 0., y_score.values)


def summary_to_latex(logit_fit, outpath):
    summary = logit_fit.summary2().as_latex()
    with open(outpath, 'w') as fd:
        fd.write(summary)

# AIC - Train/Test ROC AUC
fit_aic = fit_logit(train_aic)
train_roc_auc_aic = get_roc_auc(fit_aic, train_aic)
test_roc_auc_aic = get_roc_auc(fit_aic, test)

# BIC - Train/Test ROC AUC
fit_bic = fit_logit(train_bic)
train_roc_auc_bic = get_roc_auc(fit_bic, train_bic)
test_roc_auc_bic = get_roc_auc(fit_bic, test)

breakpoint()

# Save estimated Models as LaTeX files
outfolder = PARENT_DIR / 'meta'
summary_to_latex(fit_aic, outfolder / 'summary_logit_aic.tex')
summary_to_latex(fit_bic, outfolder / 'summary_logit_bic.tex')