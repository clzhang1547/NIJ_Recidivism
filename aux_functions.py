##################################
# Auxiliary functions needed for NIJ Challenge code
#
# chris zhang 5/11/2021
##################################
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.cluster import KMeans
from keras.layers import Input,Conv1D,MaxPooling1D,UpSampling1D

# a function to define supervision activity columns
def get_sup_act_cols():
    cols_sup_act = ['drugtests_other_positive', 'drugtests_meth_positive', 'avg_days_per_drugtest',
                    'violations_failtoreport', 'jobs_per_year', 'program_unexcusedabsences', 'residence_changes',
                    'program_attendances', 'drugtests_thc_positive', 'delinquency_reports',
                    'drugtests_cocaine_positive', 'violations_instruction', 'violations_movewithoutpermission',
                    'percent_days_employed', 'employment_exempt', 'violations_electronicmonitoring']
    return cols_sup_act

def get_prior_crime_cols():
    cols_prior_crime = ['prior_arrest_episodes_felony', 'prior_arrest_episodes_misd', 'prior_arrest_episodes_violent',
                        'prior_arrest_episodes_property', 'prior_arrest_episodes_drug', 'prior_arrest_episodes_ppviolationcharges',
                        'prior_arrest_episodes_dvcharges', 'prior_arrest_episodes_guncharges', 'prior_conviction_episodes_felony',
                        'prior_conviction_episodes_misd', 'prior_conviction_episodes_viol', 'prior_conviction_episodes_prop',
                        'prior_conviction_episodes_drug', 'prior_conviction_episodes_ppviolationcharges',
                        'prior_conviction_episodes_domesticviolencecharges', 'prior_conviction_episodes_guncharges']
    return cols_prior_crime

# a function to get cols for dummies
def get_dummy_cols(var):
    cols = ['']
    if var=='age_at_release':
        cols = ['18-22', '23-27', '28-32', '33-37', '38-42', '43-47', '48 or older']
    elif var=='education_level':
        cols = ['Less than HS diploma', 'High School Diploma', 'At least some college']
    elif var=='dependents':
        cols = ['0', '1', '2', '3 or more']
    elif var=='prison_offense':
        cols = ['Drug', 'Other', 'Property', 'Violent/Non-Sex', 'Violent/Sex']
    elif var=='prison_years':
        cols = ['Less than 1 year', '1-2 years', 'Greater than 2 to 3 years', 'More than 3 years']
    cols = [var + '_' + x for x in cols]
    return cols

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
# a function to fill in NAs, default fill-in value is random draw from valid values in column
def fill_na(col, value='random'):
    # col: a df column, pdSeries
    if True not in col.isna().value_counts().index: # no NA
        pass
    else:
        # at least 1 NA
        # fill with random value
        if value == 'random':
            col = [random.choice(col.dropna().values) if pd.isnull(x) else x for x in col]
        elif value == 'median':
            col = pd.DataFrame(col).fillna(value=col.median())
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

# a function to km-cluster a df_col, which can be purely numeric or boolean
# the ONLY non-numeric string value in df_col should be ' or more' as in '10 or more'
def get_km_subgroups(df_col):
    # df_col: a df col
    # make a copy of df_col, df_col will be unchanged
    col = df_col.copy()
    # if bool col then convert to int
    if col.dtype.name=='bool':
        col = col.astype(int)
    elif col.dtype.name=='object':
        # remove string in col values and make numeric
        col = [x.replace(' or more', '') for x in col]
        col = pd.Series(col)
        col = col.astype(int)
    # k-means fit and get labels
    km = KMeans(n_clusters=2).fit(col.values.reshape(-1, 1))
    labels = km.labels_.tolist()
    # get km-based new col
    col_name = df_col.name # for renaming km-based col
    km_col = pd.Series(labels, name=col_name)

    return km_col

# Autoencoder - 3 layers
def autoencoder(input_img):
    #encoder
    #input = k x 1 (wide and thin)
    conv1 = Conv1D(32, (3), activation='relu', padding='same')(input_img) # k x 1 x 32
    pool1 = MaxPooling1D(pool_size=2)(conv1) # k/2 x 1 x 32
    conv2 = Conv1D(64, (3), activation='relu', padding='same')(pool1) # k/2 x 1 x 64
    pool2 = MaxPooling1D(pool_size=2)(conv2) # k/2 x 1 x 64
    conv3 = Conv1D(128, (3), activation='relu', padding='same')(pool2) # k/4 x 1 x 128 (small and thick)


    #decoder
    conv4 = Conv1D(128, (3), activation='relu', padding='same')(conv3) # k/4 x 1 x 128
    up1 = UpSampling1D(2)(conv4) # k/2 x 1 x 128
    conv5 = Conv1D(64, (3), activation='relu', padding='same')(up1) # k/2 x 1 x 64
    up2 = UpSampling1D(2)(conv5) # k x 1 x 64
    decoded = Conv1D(1, (3,), activation='sigmoid', padding='same')(up2) # k x 1 x 1
    return decoded

# Autoencoder - 5 layers
def autoencoder_deep(input_img):
    # a deeper CNN with lowest phenotype dim to k/16
    # Note - k should be dividable by pool_size multiple times, as set by conv structure
    #encoder
    #input = k x 1 (wide and thin)
    conv1 = Conv1D(32, (3), activation='relu', padding='same')(input_img) # k x 1 x 32
    pool1 = MaxPooling1D(pool_size=2)(conv1) # k/2 x 1 x 32
    conv2 = Conv1D(64, (3), activation='relu', padding='same')(pool1) # k/2 x 1 x 64
    pool2 = MaxPooling1D(pool_size=2)(conv2) # k/4 x 1 x 64
    conv3 = Conv1D(128, (3), activation='relu', padding='same')(pool2) # k/4 x 1 x 128
    pool3 = MaxPooling1D(pool_size=2)(conv3) # k/8 x 1 x 128
    conv4 = Conv1D(256, (3), activation='relu', padding='same')(pool3) # k/8 x 1 x 256
    pool4 = MaxPooling1D(pool_size=2)(conv4) # k/16 x 1 x 256
    conv5 = Conv1D(512, (3), activation='relu', padding='same')(pool4) # k/16 x 1 x 512

    #decoder
    conv6 = Conv1D(512, (3), activation='relu', padding='same')(conv5) # k/16 x 1 x 512
    up1 = UpSampling1D(2)(conv6) # k/8 x 1 x 512
    conv7 = Conv1D(256, (3), activation='relu', padding='same')(up1) # k/8 x 1 x 256
    up2 = UpSampling1D(2)(conv7) # k/4 x 1 x 256
    conv8 = Conv1D(128, (3,), activation='relu', padding='same')(up2) # k/4 x 1 x 128
    up3 = UpSampling1D(2)(conv8) # k/2 x 1 x 129
    conv9 = Conv1D(64, (3,), activation='relu', padding='same')(up3) # k/2 x 1 x 64
    up4 = UpSampling1D(2)(conv9) # k x 1 x 64
    decoded = Conv1D(1, (3,), activation='sigmoid', padding='same')(up4) # k x 1 x 1
    return decoded