import pandas as pd
from sklearn import set_config
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from pdb import set_trace

from utils import PARENT_DIR

set_config(display='diagram', transform_output='pandas')

RANDOM_SEED = 8

df = pd.read_csv(PARENT_DIR / 'application_train.csv.zip', compression='zip')
df.set_index("SK_ID_CURR", inplace=True)

# Filter out rows without an annuity - to homogenise the dataset
n_without_annuity = sum(pd.isna(df.AMT_ANNUITY))
df = df.loc[~pd.isna(df.AMT_ANNUITY), ]

# After filtering, apply train/test split, to avoid data leak
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)

X = df.loc[:, df.columns != 'TARGET']
y = df['TARGET']

# numeric indices returned
train_index, test_index = list(sss.split(X, y))[0]
X_train, X_test = X.iloc[train_index, ], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Check quality of the stratified split (good, bads) - very close
train_target_split = y_train.value_counts(normalize=True)
df_target_split = df.TARGET.value_counts(normalize=True)

# Output Raw applications
train = X_train.copy()
train['TARGET'] = y_train
test = X_test.copy()
test['TARGET'] = y_test
# on-the-fly compression by extension
train.to_csv(PARENT_DIR / 'processed' / 'train_raw_apps.csv.zip')
test.to_csv(PARENT_DIR / 'processed' / 'test_raw_apps.csv.zip')

# APPLICATION-LEVEL FEATURE ENG
df["LTV"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
df["LB_Credit_Length"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
df["Credit_to_Inc"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

# NON DATA-BASED IMPUTATION
# Impute NANs in bureau applications as zeros - reasonable, since it is the mode
bureau_cols = [col for col in df.columns if 'BUREAU' in col]
for col in bureau_cols:
    df.loc[pd.isna(df[col]), col] = 0
# check - it works
missing_bureau = df[bureau_cols].isna().sum()

# Impute own car age also with zeros, when nan
# Before that, if nan, create separate dummy indicating that has no own car
df['OWN_CAR'] = pd.to_numeric(df['OWN_CAR_AGE'].isna())
df.loc[pd.isna(df['OWN_CAR_AGE']), 'OWN_CAR_AGE'] = 0

# Re-categorise Hour appr process

# Drop variables with more than 51% missing (slightly more than half)
max_missing = .51
include = ["EXT_SOURCE_1"]
number_cols_to_drop= len(df.columns[df.isna().mean() > max_missing]) - len(include)
cut_df = df.copy()
cut_df = cut_df.loc[:, df.columns[df.isna().mean() <= max_missing]]
# add back columns in `include` vector
cut_df[include] = df[include]
df = cut_df

# Rebuild splits
X = df.loc[:, df.columns != 'TARGET']
y = df['TARGET']
# numeric indices returned
X_train, X_test = X.iloc[train_index, ], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

### DATA-BASED IMPUTATION - also dealing with missing for quantitative variables

# Use metadata to categorise correctly
metadata = pd.read_csv(PARENT_DIR / "meta" / "train_summary_applications.csv", index_col=["Attribute"])
data_types = metadata["Data Type"]
pd_cols = set(df.columns)
numeric_features = [attr for attr, type in data_types.items() if type == "Quantitative" and attr in pd_cols]
categorical_features = [attr for attr, type in data_types.items() if type == "Categorical" and attr in pd_cols]
# add engineered features
numeric_features += ["LTV", "LB_Credit_Length", "Credit_to_Inc"]

# Create Column Transformer, based on numeric and categorical transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
    )

X_train_proc = preprocessor_tree.fit_transform(X_train)
X_test_proc = preprocessor_tree.transform(X_test)

# TODO: Check results. Known issue: EXT_SOURCE_1 not Quantitative?
# After checks, try to remove prefixes
set_trace()



