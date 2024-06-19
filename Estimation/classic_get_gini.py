import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from utils.utils import PARENT_DIR, TARGET
from utils.attrs import prettify_cols

# Load Data
train_aic = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_aic.csv.gz', compression="gzip")
train_bic = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_bic.csv.gz', compression="gzip")
test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")

# Prettify columns (except TARGET)
train_aic = prettify_cols(train_aic, exceptions=[TARGET])
train_bic = prettify_cols(train_bic, exceptions=[TARGET])
test = prettify_cols(test, exceptions=[TARGET])

# Replace targets
train_aic[TARGET] = train_aic[TARGET]
train_bic[TARGET] = train_bic[TARGET]
test[TARGET] = test[TARGET]


def fit_logit(train):
    # Add constants
    train = sm.add_constant(train)
    # Get Xs
    X_train = train.drop(columns=TARGET)
    # Estimate Logistic Regressions
    logit = sm.Logit(train.TARGET, X_train)
    return logit.fit()


def get_roc_auc(logit_fit, test):
    # Get X_test
    pred_names = logit_fit.params.index.tolist()[1:]  # removing intercept col
    X_test = test.loc[:, pred_names]
    # Add constant to X_test
    X_test = sm.add_constant(X_test)
    # Make predictions, obtain ROC AUC
    y_score = logit_fit.predict(X_test)
    return roc_auc_score(test.TARGET.values, y_score.values)


def get_gini(logit_fit, test):
    return 2.*get_roc_auc(logit_fit, test) - 1.


def summary_to_latex(logit_fit, outpath, caption, label):
    # Add summary of result
    res = logit_fit.summary().tables[0].as_latex_tabular()
    # Decrease table counter
    res += (
    r"""
    % Decrease due to composite table
    \addtocounter{table}{-1}
    """
    )
    # Add result detail
    detail_res = logit_fit.summary().tables[1].as_latex_tabular()
    detail_res = detail_res.replace("tabular", "longtable")
    res += detail_res

    # Add caption
    res += (
    r"""
    \begin{{center}}
        \captionsetup{{type=table}}
        \caption{{{caption}}}
        \label{{{label}}}
    \end{{center}}
    """.format(caption=caption, label=label)
    )
    with open(outpath, 'w') as fd:
        fd.write(res)

# AIC - Train/Test ROC AUC
fit_aic = fit_logit(train_aic)
train_gini_aic = get_gini(fit_aic, train_aic)
test_gini_aic = get_gini(fit_aic, test)

# BIC - Train/Test ROC AUC
fit_bic = fit_logit(train_bic)
train_gini_bic = get_gini(fit_bic, train_bic)
test_gini_bic = get_gini(fit_bic, test)

# Output results to LaTeX
gini = {
    'Train': [train_gini_aic, train_gini_bic],
    'Test': [test_gini_aic, test_gini_bic]
}
# Create the DataFrame with rows as 'AIC' and 'BIC'
df_gini = pd.DataFrame(gini, index=['AIC', 'BIC'])
df_gini = df_gini.map(lambda x: f"{x * 100:.2f}%")
# Convert the DataFrame to LaTeX
latex_output = df_gini.to_latex()
print(latex_output)

# Save estimated Models as LaTeX files
outfolder = PARENT_DIR / 'meta'
summary_to_latex(
    fit_aic, outfolder / 'summary_logit_aic.tex', 
    "Logistic Regression Estimates after Variable Selection using AIC.", "tab:logit_aic")
summary_to_latex(
    fit_bic, outfolder / 'summary_logit_bic.tex', 
    "Logistic Regression Estimates after Variable Selection using BIC.", "tab:logit_bic")