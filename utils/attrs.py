import pandas as pd
from utils import PARENT_DIR


def prettify_attrs(df: pd.DataFrame, attr_col, new_attr_col=None):
    """Function that maps the original attribute names to clean, understandable attribute names. By default, the function replaces the column original attribute names"""
    meta = pd.read_csv(PARENT_DIR / "meta" / "train_summary_applications_ext.csv")
    meta = meta[["Attribute", "Clean Attribute"]]
    meta["Attribute"] = meta["Attribute"].str.lower()

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


if __name__ == "__main__":
    # Load scorecard
    scorecard = pd.read_excel(PARENT_DIR / 'meta' / 'scorecard_bic.xlsx')
    scorecard.Attribute = scorecard.Attribute.str.upper()
    scorecard = prettify_attrs(scorecard, "Attribute")
    breakpoint()