##################################
# NIJ Recidivism Challenge
# Year 2 forecast
# chris zhang 6/14/2021
##################################
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
import sklearn.preprocessing
import xgboost
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import random
from time import time
from collections import OrderedDict
import json

from aux_functions import *

#######################
# Data Set Up
#######################
# Read in data
def get_df_for_forecast(fp_in):
    #d = read_and_clean_raw('./data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
    d = read_and_clean_raw(fp_in)

    # Recode all ordinal features with numerical-like values
    cols_ordinal = get_ordinal_cols()
    for c in cols_ordinal:
        d[c] = convert_ordinal_to_numeric(d[c])

    # Fill in NAs
    d = get_df_na_filled(d)

    # Convert two-class cols to numeric
    # the resulting df is good for cat-col based method such as LGBM (no need to One Hot Encode)
    d = convert_two_class_cols_to_numeric(d)
    dc = d.copy() # a 'cat-col' version of d before One Hot Encoding (for LGBM use only)

    # Get categorical cols (for LGBM use only)
    bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(dc)
    cols_no_enc = get_cols_no_encode()
    cols_ordinal = get_ordinal_cols()
    cat_cols = set(dc.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) \
               - set(cols_no_enc) - set(cols_ordinal)

    # One Hot Encoding
    d = one_hot_encoding(d)

    # Make cols for recidivism path
    dct_reci_path = get_dict_risky_prior_crime_types()
    for k, v in dct_reci_path.items():
        # k = current prison_offense
        # v = list of relevant prior crime types
        d['path_%s' % k] = 0
        for p in v: # p = prior crime type
            # binary path
            # d['path_%s' % k] = np.where((d['prison_offense_%s' % k]==1) &
            #                             (d['prior_arrest_episodes_%s' % p]>2), 1, d['path_%s' % k])
            # alternatively, path intensity
            d['path_%s' % k] += d['prison_offense_%s' % k] * d['prior_arrest_episodes_%s' % p]
    return d

d = get_df_for_forecast('./data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
dt = get_df_for_forecast('./data/nij/NIJ_s_Recidivism_Challenge_Test_Dataset2.csv')
dt = dt.rename(columns={'prior_conviction_episodes_domesticviolencecharges': 'prior_conviction_episodes_dvcharges'})
# Define cols for model training's X, y
cols_ys = ['recidivism_within_3years', 'recidivism_arrest_year1', 'recidivism_arrest_year2', 'recidivism_arrest_year3']
# supervision activity cols, to be excl. from year 1 model
cols_sup_act = get_sup_act_cols()
# for d (one hot encoding version)
cols_X = [x for x in d.columns if x not in ['id']+ cols_ys]
cols_X1 = [x for x in d.columns if (x not in ['id'] + cols_ys) and
           (x.replace('_' + x.split('_')[-1], '') not in cols_sup_act) and
           (x not in cols_sup_act)] # remove sup act cols for year 1 features
col_y = 'recidivism_arrest_year2'

########################################################
# Year 2 Model Training with Tuned Parameters
########################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pprint import pprint
# initialize phat_out
phat_out = pd.DataFrame()

# clf for PCA
clf = LogisticRegression(max_iter=5000)

# Subgroup: path_Violent/Sex > 0
path_xvar, subgroup_bool = 'path_Violent/Sex', True
df = d[(d[path_xvar]>0)==subgroup_bool]
y = df[df['recidivism_arrest_year1'] == 0][col_y]
X = df[df['recidivism_arrest_year1'] == 0][cols_X]

# test data
dft = dt[(dt[path_xvar]>0)==subgroup_bool]
Xt = dft[cols_X]
# Model
pipe = Pipeline([("scale", StandardScaler()),
                 ("reduce_dims", PCA(n_components=0.25)),
                 ('model', clf)])
# quick validation
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
dct_score = cross_validate(pipe, X, y, cv=5, scoring=scores)
print('Average Score [Subgroup 1 CV5. Total N = %s]:' % len(Xt))
pprint({k: round(v.mean(), 6) for k, v in dct_score.items()})
# fit and predict
pipe.fit(X, y)
print('Final Subgroup 1 PCA n_components = %s' % pipe.__dict__['steps'][1][1].n_components_)
phat = pipe.predict_proba(Xt)
phat = pd.Series([x[1] for x in phat], name='Probability')
id = pd.Series([int(x) for x in dft['id'].values], name='ID', dtype=int)
phat = pd.DataFrame([id, phat]).T
# append subgroup results to master phat_out
phat_out = phat_out.append(phat)

# Subgroup: path_Property = 0
path_xvar = ['path_Violent/Sex', 'path_Property']
subgroup_bool = [False, False]

df = d[((d[path_xvar[0]]>0)==subgroup_bool[0]) &
       ((d[path_xvar[1]]>0)==subgroup_bool[1])]
y = df[df['recidivism_arrest_year1'] == 0][col_y]
X = df[df['recidivism_arrest_year1'] == 0][cols_X]

# test data
dft = dt[((dt[path_xvar[0]]>0)==subgroup_bool[0]) &
       ((dt[path_xvar[1]]>0)==subgroup_bool[1])]

Xt = dft[cols_X]
# Model
pipe = Pipeline([("scale", StandardScaler()),
                 ("reduce_dims", PCA(n_components=0.75)),
                 ('model', clf)])
# quick validation
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
dct_score = cross_validate(pipe, X, y, cv=5, scoring=scores)
print('Average Score [Subgroup 2 CV5. Total N = %s]:' % len(Xt))
pprint({k: round(v.mean(), 6) for k, v in dct_score.items()})
# fit and predict
pipe.fit(X, y)
print('Final Subgroup 2 PCA n_components = %s' % pipe.__dict__['steps'][1][1].n_components_)
phat = pipe.predict_proba(Xt)
phat = pd.Series([x[1] for x in phat], name='Probability')
id = pd.Series([int(x) for x in dft['id'].values], name='ID', dtype=int)
phat = pd.DataFrame([id, phat]).T
# append subgroup results to master phat_out
phat_out = phat_out.append(phat)

# Subgroup: rest
# Use RF n_estimator=400
path_xvar = ['path_Violent/Sex', 'path_Property']
subgroup_bool = [False, True]

df = d[((d[path_xvar[0]]>0)==subgroup_bool[0]) &
       ((d[path_xvar[1]]>0)==subgroup_bool[1])]
y = df[df['recidivism_arrest_year1'] == 0][col_y]
X = df[df['recidivism_arrest_year1'] == 0][cols_X]

# test data
dft = dt[((dt[path_xvar[0]]>0)==subgroup_bool[0]) &
       ((dt[path_xvar[1]]>0)==subgroup_bool[1])]

Xt = dft[cols_X]
# Model
clf = RandomForestClassifier(n_estimators=400)
# quick validation
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
dct_score = cross_validate(clf, X, y, cv=5, scoring=scores)
print('Average Score [Subgroup 3 CV5. Total N = %s]:' % len(Xt))
pprint({k: round(v.mean(), 6) for k, v in dct_score.items()})
# fit and predict
clf.fit(X, y)
phat = clf.predict_proba(Xt)
phat = pd.Series([x[1] for x in phat], name='Probability')
id = pd.Series([int(x) for x in dft['id'].values], name='ID', dtype=int)
phat = pd.DataFrame([id, phat]).T
# append subgroup results to master phat_out
phat_out = phat_out.append(phat)

# Export phat_out
phat_out = phat_out.sort_values(by='ID')
phat_out['Probability'] = [round(x, 4) for x in phat_out['Probability']]
phat_out.to_csv('./output/year_2_forecast/year_2_forecast.csv', index=False)
phat_out.to_excel('./_management/_submission/Year2/AMERICAN INSTITUTES FOR RESEARCH_2YearForecast.xlsx', index=False)

###########################

df = d.copy()
y = df[df['recidivism_arrest_year1'] == 0][col_y]
X = df[df['recidivism_arrest_year1'] == 0][cols_X]
# quick validation
clf = LogisticRegression(max_iter=5000)
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
dct_score = cross_validate(clf, X, y, cv=5, scoring=scores)
print('Average Score [Subgroup 3 CV5. Total N = %s]:' % len(Xt))
pprint({k: round(v.mean(), 6) for k, v in dct_score.items()})


