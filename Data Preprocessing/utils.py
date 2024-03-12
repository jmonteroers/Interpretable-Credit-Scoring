from pathlib import Path

PARENT_DIR = Path(__file__).absolute().parents[2] / 'Data' / 'Home Credit'
CURRENT_ID = "SK_ID_CURR"
PREV_ID = "SK_ID_PREV"
BUREAU_ID = "SK_ID_BUREAU"

# Just for exploration purposes
SAMPLE_ID_CURR = 350419


# Util functions
def add_count(df, aux_df, aux_id, new_colname, subset=None, main_id=CURRENT_ID):
    """Returns new df"""
    if subset is not None:
        aux_df = aux_df.loc[subset]
    df_ext = df.merge(aux_df, on=main_id, how="left")
    counts = df_ext.groupby(main_id)[aux_id].count().reset_index()
    counts.rename(columns={aux_id: new_colname}, inplace=True)
    df = df.merge(counts, on=main_id, how="left")
    df[new_colname].fillna(0, inplace=True)
    return df


def add_exposure(df, aux_df, subset, exp_col, new_col, main_id=CURRENT_ID):
    aux_df = aux_df.loc[subset].copy()
    # if na in exposure column, assume 0
    aux_df[exp_col].fillna(0, inplace=True)
    df_ext = df.merge(aux_df, on=main_id, how="left")
    exposure = df_ext.groupby(main_id)[exp_col].sum().reset_index()
    exposure.rename(columns={exp_col: new_col}, inplace=True)
    return df.merge(exposure, on=main_id, how="left")


def add_last_credit(df, aux_df, age_col, new_colname, na_imp=None, main_id=CURRENT_ID):
    """Returns new df"""
    df_ext = df.merge(aux_df, on=main_id, how="left")
    last_credit = df_ext.groupby(main_id)[age_col].max().reset_index()
    last_credit.rename(columns={age_col: new_colname}, inplace=True)
    df = df.merge(last_credit, on=main_id, how="left")
    if na_imp is not None:
        df[new_colname].fillna(na_imp, inplace=True)
    return df