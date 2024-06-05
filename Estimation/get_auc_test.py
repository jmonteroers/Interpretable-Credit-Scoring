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
train_aic[TARGET] = train_aic[TARGET] == 0.
train_bic[TARGET] = train_bic[TARGET] == 0.
test[TARGET] = test[TARGET] == 0.


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
    return roc_auc_score(test.TARGET.values == 0., y_score.values)


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
train_roc_auc_aic = get_roc_auc(fit_aic, train_aic)
test_roc_auc_aic = get_roc_auc(fit_aic, test)

# BIC - Train/Test ROC AUC
fit_bic = fit_logit(train_bic)
train_roc_auc_bic = get_roc_auc(fit_bic, train_bic)
test_roc_auc_bic = get_roc_auc(fit_bic, test)

# Output results to LaTeX
roc_auc = {
    'Train': [train_roc_auc_aic, train_roc_auc_bic],
    'Test': [test_roc_auc_aic, test_roc_auc_bic]
}
# Create the DataFrame with rows as 'AIC' and 'BIC'
df_roc_auc = pd.DataFrame(roc_auc, index=['AIC', 'BIC'])
# Convert the DataFrame to LaTeX
latex_output = df_roc_auc.to_latex()
print(latex_output)

# Save estimated Models as LaTeX files
outfolder = PARENT_DIR / 'meta'
summary_to_latex(fit_aic, outfolder / 'summary_logit_aic.tex', "This love has taken its toll on me", "tab:love")
summary_to_latex(fit_bic, outfolder / 'summary_logit_bic.tex', "Love to hate her, hate to love her", "tab:love-hate")