import pandas as pd
from utils import PARENT_DIR


def load_meta():
    meta = pd.read_csv(PARENT_DIR / "meta" / "train_summary_applications_ext.csv")
    meta = meta[["Attribute", "Clean Attribute"]]
    meta["Attribute"] = meta["Attribute"].str.lower()
    return meta


def prettify_attrs(df: pd.DataFrame, attr_col, new_attr_col=None):
    """Function that maps the original attribute names to clean, understandable attribute names. By default, the function replaces the column original attribute names"""
    meta = load_meta()

    # Merge
    # NOTE: Assuming 'Merge Key' not being used as a column
    df["Merge Key"] = df[attr_col].str.lower()
    df = df.merge(meta, how="left", left_on="Merge Key", right_on="Attribute")

    # Give clean attributes desired name
    new_attr_col = attr_col if new_attr_col is None else new_attr_col
    df[new_attr_col] = df["Clean Attribute"]
    df.drop(columns="Clean Attribute", inplace=True)

    # Drop Merge Key
    df.drop(columns="Merge Key", inplace=True)

    return df


def prettify_cols(df: pd.DataFrame, exceptions=None):
    """Maps the original attributes as columns to clean, understandable names. Acts in place"""
    # Build exceptions, not to be mapped
    exceptions = [] if exceptions is None else exceptions

    # Load mapping
    meta = load_meta()
    meta = meta.set_index("Attribute")

    # Traverse columns, replacing by clean value when key available
    new_columns = []
    for col in df.columns.tolist():
        if col in exceptions or col.lower() not in meta.index:
            new_columns.append(col)
        else:
            new_columns.append(meta.loc[col.lower(), "Clean Attribute"])
    df.columns = new_columns

    return df


if __name__ == "__main__":
    # Prettify attributes
    # Load scorecard
    scorecard = pd.read_excel(PARENT_DIR / 'meta' / 'scorecard_bic.xlsx')
    scorecard.Attribute = scorecard.Attribute.str.upper()
    scorecard = prettify_attrs(scorecard, "Attribute")

    # Prettify columns
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_bic_npos.csv.gz', compression='gzip')
    train_clean = prettify_cols(train)
    breakpoint()