##################################
# Auxiliary functions needed for NIJ Challenge code
#
# chris zhang 5/11/2021
##################################
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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

# a function to randomly set 0s to 2d array
def set_random_elements(array, prop, fill=0):
    # array = 2d array with elements in list (so actually 3d), prop = proprtion between 0 to 1, fill = value to fill in
    fill_size = int(array.size*prop) # number of fills to perform
    random_idxs0 = np.random.randint(0, array.shape[0], size=fill_size)
    random_idxs1 = np.random.randint(0, array.shape[1], size=fill_size)
    filled_array = array.copy()
    filled_array[random_idxs0, random_idxs1] = fill
    return filled_array

def set_random_nonzero_elements(array, prop, fill=0):
    # array = 2d array with elements in list (so actually 3d), prop = proprtion between 0 to 1, fill = value to fill in
    idxs_nonzero = np.nonzero(array.reshape(-1, array.shape[1]))
    idxs_nonzero = list(zip(idxs_nonzero[0], idxs_nonzero[1]))
    fill_size = int(len(idxs_nonzero)*prop) # number of fills to perform
    random_idxs_spots = np.random.choice(len(idxs_nonzero), size=fill_size) # location of chosen index tuples in nonzero list

    random_idxs = np.array(idxs_nonzero)[random_idxs_spots] # chosen indices to be filled
    # random indices along both axes
    random_idxs0 = random_idxs[:, 0]
    random_idxs1 = random_idxs[:, 1]

    filled_array = array.copy()
    filled_array[random_idxs0, random_idxs1] = fill
    return filled_array