import pandas as pd
from sklearn import set_config
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from pdb import set_trace

from add_features.utils import PARENT_DIR
from add_features.main import add_features

set_config(display='diagram', transform_output='pandas')

RANDOM_SEED = 8

def create_train_test_split(parent_dir=PARENT_DIR, test_size=0.3):
    """Applies some initial filtering, splits into train and test. Takes as input original application_train.csv.zip, saves as output train/test_raw_apps.csv.zip"""
    df = pd.read_csv(parent_dir / 'application_train.csv.zip', compression='zip')
    df.set_index("SK_ID_CURR", inplace=True)

    # Filter out rows without an annuity - to homogenise the dataset
    n_without_annuity = sum(pd.isna(df.AMT_ANNUITY))
    print(f"Observations without annuity dropped {n_without_annuity}")
    df = df.loc[~pd.isna(df.AMT_ANNUITY), ]

    # After filtering, apply train/test split, to avoid data leak
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)

    X = df.loc[:, df.columns != 'TARGET']
    y = df['TARGET']

    # numeric indices are returned
    train_index, test_index = list(sss.split(X, y))[0]

    # Output Raw applications
    train, test = df.iloc[train_index, ], df.iloc[test_index, ]

    # Check quality of the stratified split (good, bads) - very close
    # train_target_split = train.TARGET.value_counts(normalize=True)
    # test_target_split = test.TARGET.value_counts(normalize=True)

    # on-the-fly compression by extension
    train.to_csv(parent_dir / 'processed' / 'train_raw_apps.csv.zip')
    test.to_csv(parent_dir / 'processed' / 'test_raw_apps.csv.zip')

    return df, train_index, test_index





if __name__ == "__main__":
    # Note - this is saving initial output
    df, train_idx, test_idx = create_train_test_split()

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
    # missing_bureau = df[bureau_cols].isna().sum()

    # Impute own car age also with zeros, when nan
    # Before that, if nan, create separate dummy indicating that has no own car
    df['OWN_CAR'] = pd.to_numeric(df['OWN_CAR_AGE'].isna())
    df.loc[pd.isna(df['OWN_CAR_AGE']), 'OWN_CAR_AGE'] = 0

    # Re-categorise Hour appr process
    # Define function to map hour to time of day
    def map_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 24:
            return 'Evening'
        else:
            return 'Night'

    # Apply the function to create a new column
    df['DAYTIME_PROCESS_START'] = df['HOUR_APPR_PROCESS_START'].apply(map_time_of_day)
    df.drop(columns='HOUR_APPR_PROCESS_START', inplace=True)

    # Drop variables with more than 51% missing (slightly more than half)
    max_missing = .51
    include = ["EXT_SOURCE_1"]
    number_cols_to_drop= len(df.columns[df.isna().mean() > max_missing]) - len(include)
    cut_df = df.copy()
    cut_df = cut_df.loc[:, df.columns[df.isna().mean() <= max_missing]]
    # add back columns in `include` vector
    cut_df[include] = df[include]
    df = cut_df

    # Engineer features
    df = add_features(df)
    # Check - do engineered features have any missing?
    print(df.iloc[:, -20:].describe())

    # Build train, test splits - Save output
    train = df.iloc[train_idx, ]
    test = df.iloc[test_idx, ]
    # on-the-fly compression by extension
    train.to_csv(PARENT_DIR / 'processed' / 'train_apps_ext.csv.zip')
    test.to_csv(PARENT_DIR / 'processed' / 'test_apps_ext.csv.zip')


    ### DATA-BASED IMPUTATION - also dealing with missing for quantitative variables

    # Split into X, y
    X = df.loc[:, df.columns != 'TARGET']
    X_train, y_train = train.loc[:, train.columns != 'TARGET'], train.TARGET
    X_test, y_test = test.loc[:, test.columns != 'TARGET'], test.TARGET

    # Use metadata to categorise correctly
    metadata = pd.read_csv(PARENT_DIR / "meta" / "train_summary_applications.csv", index_col=["Attribute"])
    data_types = metadata["Data Type"]
    pd_cols = set(df.columns)
    numeric_features = [attr for attr, type in data_types.items() if type == "Quantitative" and attr in pd_cols]
    categorical_features = [attr for attr, type in data_types.items() if type == "Categorical" and attr in pd_cols]
    # NOTE: applicant-level engineered features checked to have no missing. Otherwise, add here as needed
    app_numeric_feats = ["LTV", "LB_Credit_Length", "Credit_to_Inc"]
    numeric_features += app_numeric_feats
    # add daytime to categorical
    categorical_features += ['DAYTIME_PROCESS_START']

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



