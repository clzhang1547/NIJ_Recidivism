##################################
# Auxiliary functions needed for NIJ Challenge code
#
# chris zhang 5/11/2021
##################################
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

# a function to read and clean up raw data
def read_and_clean_raw(fp):
    df = pd.read_csv(fp)
    # Rename varnames, set all varname to lower
    dct_rename = {
        '_v1': 'Prior_Arrest_Episodes_PPViolationCharges',
        '_v2': 'Prior_Conviction_Episodes_PPViolationCharges',
        '_v3': 'Prior_Conviction_Episodes_DomesticViolenceCharges',
        '_v4': 'Prior_Conviction_Episodes_GunCharges',
    }
    df = df.rename(columns=dct_rename)
    df.columns = [x.lower() for x in df.columns]
    return df
# a function to fill in NAs using random draw from valid values in column
def fill_na(col):
    # col: a df column, pdSeries
    if True not in col.isna().value_counts().index: # no NA
        pass
    else: # at least 1 NA
        col = [random.choice(col.dropna().values) if pd.isnull(x) else x for x in col]
    return col

# a function to perform One Hot Encoding
def one_hot_encoding(X, col_names):
    # X: a 2d array of feature values
    # col_names: column names of all features in X
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    out = enc.transform(X).toarray()
    cols = enc.get_feature_names(col_names)
    df = pd.DataFrame(out, columns=cols)

    return df

# a function to get boolean columns
def get_bool_binary_cols(df):
    bool_cols = [c for c in df.columns
                 if df[c].dropna().value_counts().index.isin([True, False]).all()]
    binary_cols = [c for c in df.columns
                 if df[c].dropna().value_counts().index.isin([0, 1]).all()]
    # two class cols contain two values other than True/False and 0/1, e.g. gender=male, female
    two_class_cols = [c for c in df.columns
                 if len(df[c].dropna().value_counts().index)==2]
    two_class_cols = list(set(two_class_cols) - set(bool_cols) - set(binary_cols))
    return (bool_cols, binary_cols, two_class_cols)