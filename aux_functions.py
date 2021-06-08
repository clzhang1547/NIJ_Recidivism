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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, \
    r2_score, mean_squared_error
import math
from string import digits
import mord

# a function to read and clean up raw data
def read_and_clean_raw(fp):
    df = pd.read_csv(fp)
    # Rename varnames, set all varname to lower
    dct_rename = {
        '_v1': 'Prior_Arrest_Episodes_PPViolationCharges',
        '_v2': 'Prior_Conviction_Episodes_PPViolationCharges',
        '_v3': 'Prior_Conviction_Episodes_DVCharges',
        '_v4': 'Prior_Conviction_Episodes_GunCharges',
        'Prior_Arrest_Episodes_Violent': 'Prior_Arrest_Episodes_Viol',
        'Prior_Arrest_Episodes_Property': 'Prior_Arrest_Episodes_Prop'  # same with prior_conviction short name
    }
    df = df.rename(columns=dct_rename)
    df.columns = [x.lower() for x in df.columns]
    # check dtype, set puma to str
    #print(d.dtypes)
    df['residence_puma'] = pd.Series(df['residence_puma'], dtype='str')
    return df

# a function to get ordinal columns
def get_ordinal_cols():
    cols_ordinal = ['age_at_release']
    cols_ordinal += ['dependents', 'prison_years',
                     'prior_arrest_episodes_felony', 'prior_arrest_episodes_misd', 'prior_arrest_episodes_viol',
                     'prior_arrest_episodes_prop', 'prior_arrest_episodes_drug',
                     'prior_arrest_episodes_ppviolationcharges', 'prior_conviction_episodes_felony',
                     'prior_conviction_episodes_misd', 'prior_conviction_episodes_prop',
                     'prior_conviction_episodes_drug']
    cols_ordinal += ['delinquency_reports', 'program_attendances', 'program_unexcusedabsences', 'residence_changes']
    return cols_ordinal

# a function to convert ordinal features to numeric values
def convert_ordinal_to_numeric(d_col):
    # d_col a pd Series col
    # col string labels must be ordered ordinally - e.g. '0', '1', '2', '3 or more'
    # after removing ' or more' string, must be purely numerical values ready for ordering

    # Remove the non-numerical characters from string values of feature
    # e.g. 'More than 3 years'>>'3', '10 or more'>>'10'
    col = d_col.name
    d_col = [''.join(z for z in x if z in digits) if isinstance(x, str) else x for x in d_col]
    d_col = pd.Series(d_col, name=col)
    col_labels = d_col.value_counts().sort_index().index
    d_col = d_col.replace(dict(zip(col_labels, range(len(col_labels)))))
    return d_col

# a function to fill in NAs for a column, default fill-in value is random draw from valid values in column
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

# a function to fill NAs for a df
def get_df_na_filled(d):
    # d: input df
    # Fill in NAs
    print(d.isna().sum())
    na_count = d.isna().sum()
    na_cols = na_count[na_count>0].index
    for c in na_cols:
        d[c] = fill_na(d[c])

    return d
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

# a function to convert two class to numeric
def convert_two_class_cols_to_numeric(d):
    # d: input df
    # e.g. bool_cols: True/False, binary_cols: 0/1, two_class_cols: 'male'/'female'
    # find boolean columns, then set False, True = 0, 1 for bool cols
    bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
    for c in bool_cols:
        d[c] = [int(x) if not np.isnan(x) else np.nan for x in d[c]]
    print(two_class_cols) # manually encode two class cols
    d['female'] = np.where(d['gender']=='F', 1, 0)
    d['black'] = np.where(d['race']=='BLACK', 1, 0)
    d = d.drop(columns=two_class_cols)
    return d

# a function to get cols not to be one hot encoded
def get_cols_no_encode():
    # cols not to be encoded
    # incl. ID and raw numeric cols
    cols_no_enc = ['id', 'supervision_risk_score_first',
                   'avg_days_per_drugtest', 'drugtests_thc_positive', 'drugtests_cocaine_positive',
                   'drugtests_meth_positive', 'drugtests_other_positive', 'percent_days_employed', 'jobs_per_year',]
    return cols_no_enc


# a function to perform One Hot Encoding
def one_hot_encoding(d):
    # d: input df
    # - Note: d must have all two-class cols converted to numeric
    # One Hot Encoding
    # get binary_cols with female, black so excl. from cat_cols
    bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
    # cols not to be encoded
    # incl. ID and raw numeric cols
    cols_no_enc = get_cols_no_encode()
    cols_ordinal = get_ordinal_cols()
    # define categorical (3+ cats) cols
    cat_cols = set(d.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) \
               - set(cols_no_enc) - set(cols_ordinal)
    # one hot encoding
    # Note 1: set drop_first=True for purely linear logit (statsmodel), set to False for ML/tree methods
    # Note 2: cols with diff dtype will not be encoded, so puma must be converted to str first
    dummies = pd.get_dummies(d[cat_cols], drop_first=False)
    d = d.join(dummies)
    d = d.drop(columns=list(cat_cols) + two_class_cols)
    return d

# a function to define supervision activity columns
def get_sup_act_cols():
    cols_sup_act = ['drugtests_other_positive', 'drugtests_meth_positive', 'avg_days_per_drugtest',
                    'violations_failtoreport', 'jobs_per_year', 'program_unexcusedabsences', 'residence_changes',
                    'program_attendances', 'drugtests_thc_positive', 'delinquency_reports',
                    'drugtests_cocaine_positive', 'violations_instruction', 'violations_movewithoutpermission',
                    'percent_days_employed', 'employment_exempt', 'violations_electronicmonitoring']
    return cols_sup_act

def get_prior_crime_cols():
    cols_prior_crime = ['prior_arrest_episodes_felony', 'prior_arrest_episodes_misd', 'prior_arrest_episodes_viol',
                        'prior_arrest_episodes_prop', 'prior_arrest_episodes_drug', 'prior_arrest_episodes_ppviolationcharges',
                        'prior_arrest_episodes_dvcharges', 'prior_arrest_episodes_guncharges', 'prior_conviction_episodes_felony',
                        'prior_conviction_episodes_misd', 'prior_conviction_episodes_viol', 'prior_conviction_episodes_prop',
                        'prior_conviction_episodes_drug', 'prior_conviction_episodes_ppviolationcharges',
                        'prior_conviction_episodes_dvcharges', 'prior_conviction_episodes_guncharges']
    return cols_prior_crime

# a function to get dict from latest GA prison_offense to relevant prior GA crime type suggesting redicivism risk
def get_dict_risky_prior_crime_types():
    dct = {}
    dct['Violent/Sex'] = ['felony', 'misd', 'viol', 'ppviolationcharges',
                          'dvcharges',]
    dct['Violent/Non-Sex'] = ['felony', 'misd', 'viol', 'ppviolationcharges',
                          'dvcharges', 'guncharges']
    dct['Property'] = ['prop', 'drug', 'ppviolationcharges',]
    dct['Drug'] = ['drug', 'ppviolationcharges', 'guncharges']
    dct['Other'] = ['ppviolationcharges']
    return dct

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



# # a function to perform One Hot Encoding
# def one_hot_encoding(X, col_names):
#     # X: a 2d array of feature values
#     # col_names: column names of all features in X
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc.fit(X)
#     out = enc.transform(X).toarray()
#     cols = enc.get_feature_names(col_names)
#     df = pd.DataFrame(out, columns=cols)
#
#     return df



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

# Imputation - get expected supervision activities
def get_expected_sup_act_cols(xvars_yr1, sup_act):
    # xvars_yr1: df of NIJ features (one-hot encoded) without 16 supervision activity cols
    # sup_act: df of 16 sup_act cols
    # init output df
    exp_sup_act = pd.DataFrame([])
    cols_sup_act = get_sup_act_cols()
    for c in cols_sup_act:
        if c in ['violations_electronicmonitoring', 'violations_failtoreport', 'violations_instruction',
                 'violations_movewithoutpermission', 'employment_exempt']:
            clf =  LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs')
            clf.fit(xvars_yr1, sup_act[c])
            exp_sup_act['_exp_%s' % c] = clf.predict(xvars_yr1)
        elif c in ['drugtests_cocaine_positive', 'drugtests_meth_positive', 'drugtests_other_positive',
                   'drugtests_thc_positive', 'percent_days_employed', 'jobs_per_year']:
            clf = LinearRegression()
            clf.fit(xvars_yr1, sup_act[c])
            exp_sup_act['_exp_%s' % c] = clf.predict(xvars_yr1)
            if c=='jobs_per_year': # c bound by 0+
                exp_sup_act['_exp_%s' % c] = [max(x, 0) for x in exp_sup_act['_exp_%s' % c]]
            else: # c bound by [0,1]
                exp_sup_act['_exp_%s' % c] = [min(max(x, 0), 1) for x in exp_sup_act['_exp_%s' % c]]
        elif c in ['avg_days_per_drugtest']: # use log-OLS for highly skewed outvar
            clf = LinearRegression()
            clf.fit(xvars_yr1, [math.log(x) for x in sup_act[c]])
            exp_sup_act['_exp_%s' % c] = [math.exp(x) for x in clf.predict(xvars_yr1)]
        elif c in ['delinquency_reports', 'program_attendances', 'program_unexcusedabsences',
                   'residence_changes']:
            clf = mord.LogisticAT().fit(xvars_yr1, sup_act[c])
            exp_sup_act['_exp_%s' % c] = clf.predict(xvars_yr1)

        # evaluate pred vs truth
        scores = {}
        if clf.__class__.__name__=='LogisticRegression':
            scores['accuracy'] = accuracy_score(sup_act[c], exp_sup_act['_exp_%s' % c])
            scores['precision'] = precision_score(sup_act[c], exp_sup_act['_exp_%s' % c])
            scores['recall'] = recall_score(sup_act[c], exp_sup_act['_exp_%s' % c])
            scores['f1'] = f1_score(sup_act[c], exp_sup_act['_exp_%s' % c])
        elif clf.__class__.__name__ == 'LinearRegression':
            scores['r2'] = r2_score(sup_act[c], exp_sup_act['_exp_%s' % c])
            scores['neg_mean_squared_error'] = mean_squared_error(sup_act[c], exp_sup_act['_exp_%s' % c])
        elif clf.__class__.__name__ == 'LogisticAT':
            scores['accuracy'] = accuracy_score(sup_act[c], exp_sup_act['_exp_%s' % c])

        print('Supervision Activity col = %s' % c)
        print('Imputation scores = %s\n' % scores)


    return exp_sup_act

