import pandas as pd


def concat_train_test(train, test):
    common_cols = train.columns.intersection(test.columns).tolist()
    train = train[common_cols]
    test = test[common_cols]
    train["Dataset"] = "Train"
    test["Dataset"] = "Test"
    concat_df = pd.concat([train, test], axis=0)
    return concat_df