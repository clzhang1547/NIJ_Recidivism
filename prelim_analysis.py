##################################
# NIJ Recidivism Challenge
# preliminary analysis
# chris zhang 5/14/2021
##################################

# TODO [done]: ues categorical cols for XGBoost
# TODO: for minor class pick best thre for prediction, based on maximizing alt metrics (e.g. G-mean, Youden)
# https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
# TODO [done]: baseline: random, one class, L2 logit
# TODO [done]: neural network
# TODO: add in geo chars, police dispatching


import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
import xgboost
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import random
from time import time

from aux_functions import *

d = pd.read_csv('./data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
dt = pd.read_csv('./data/nij/NIJ_s_Recidivism_Challenge_Test_Dataset1.csv')

# Rename varnames, set all varname to lower
dct_rename = {
    '_v1':'Prior_Arrest_Episodes_PPViolationCharges',
    '_v2': 'Prior_Conviction_Episodes_PPViolationCharges',
    '_v3': 'Prior_Conviction_Episodes_DomesticViolenceCharges',
    '_v4': 'Prior_Conviction_Episodes_GunCharges',
}
d = d.rename(columns=dct_rename)
dt = dt.rename(columns=dct_rename)
d.columns = [x.lower() for x in d.columns]
dt.columns = [x.lower() for x in dt.columns]
cols_d = set(d.columns)
cols_dt = set(dt.columns)

# Fill in NAs
print(d.isna().sum())
na_count = d.isna().sum()
na_cols = na_count[na_count>0].index
for c in na_cols:
    d[c] = fill_na(d[c])
# check dtype, set puma to int
#print(d.dtypes)
d['residence_puma'] = pd.Series(d['residence_puma'], dtype='str')

# One Hot Encoding
# find boolean columns - set False, True = 0, 1 for bool cols
bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
d[bool_cols] = d[bool_cols].astype(int)
print(two_class_cols) # manually encode two class cols
d['female'] = np.where(d['gender']=='F', 1, 0)
d['black'] = np.where(d['race']=='BLACK', 1, 0)
# cols not to be encoded
cols_no_enc = ['id', 'supervision_risk_score_first',
               'avg_days_per_drugtest', 'drugtests_thc_positive', 'drugtests_cocaine_positive',
               'drugtests_meth_positive', 'drugtests_other_positive', 'percent_days_employed', 'jobs_per_year',]
# define categorica (3+ cats) cols
cat_cols = cols_d - set(bool_cols) - set(binary_cols) - set(two_class_cols) - set(cols_no_enc)
# set aside a df with original cat cols (for XGBoost)
dc = d.copy()
dc = dc.drop(columns=two_class_cols)
# for master d, do one hot encoding for cat cols (for sk-learn models)
# one hot encoding
# Note 1: set drop_first=False for purely linear logit (statsmodel)
# Note 2: cols with diff dtype will not be encoded, so puma must be converted to str first
dummies = pd.get_dummies(d[cat_cols], drop_first=False)
d = d.join(dummies)
d = d.drop(columns=list(cat_cols) + two_class_cols)

# Define cols for model training's X, y
cols_ys = ['recidivism_within_3years', 'recidivism_arrest_year1', 'recidivism_arrest_year2', 'recidivism_arrest_year3']
# supervision activity cols, to be excl. from year 1 model
cols_sup_act = ['drugtests_other_positive', 'drugtests_meth_positive', 'avg_days_per_drugtest',
                'violations_failtoreport', 'jobs_per_year', 'program_unexcusedabsences', 'residence_changes',
                'program_attendances', 'drugtests_thc_positive', 'delinquency_reports',
                'drugtests_cocaine_positive', 'violations_instruction', 'violations_movewithoutpermission',
                'percent_days_employed', 'employment_exempt', 'violations_electronicmonitoring']
# for d (one hot encoding versoin)
cols_X = [x for x in d.columns if x not in ['id']+ cols_ys]
cols_X1 = [x for x in d.columns if x not in ['id'] + cols_ys + cols_sup_act] # remove sup act cols for year 1 features
# for dc (cat col version), convert cat cols to integers so readable by lgb
# exceptions: puma (keep orig codes)
# TODO: keep a mapping between cats and codes
for c in [x for x in cat_cols if x not in ['residence_puma']]:
    dc[c] = pd.Categorical(dc[c]).codes
cols_Xc = [x for x in dc.columns if x not in ['id']+ cols_ys]
cols_X1c = [x for x in dc.columns if x not in ['id'] + cols_ys + cols_sup_act] # remove sup act cols for year 1 features
# cat_cols include the 16 supervision cols, remove for Year 1
cat_cols_1 = [x for x in cat_cols if x not in cols_sup_act]
# Tree methods
'''
CV 5 results
Dummy (Major Class)
Year 1: {'roc_auc': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.7017, 'neg_brier_score': -0.2983}
Year 2: {'roc_auc': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.7429, 'neg_brier_score': -0.2571}
Year 3: {'roc_auc': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.8094, 'neg_brier_score': -0.1906}

Logit (L2)
Year 1: {'roc_auc': 0.7403, 'f1': 0.4243, 'precision': 0.62, 'recall': 0.3253, 'accuracy': 0.7381, 'neg_brier_score': -0.1775}
Year 2: {'roc_auc': 0.6888, 'f1': 0.2159, 'precision': 0.4989, 'recall': 0.1386, 'accuracy': 0.7426, 'neg_brier_score': -0.1762}
Year 3: {'roc_auc': 0.6568, 'f1': 0.0545, 'precision': 0.3689, 'recall': 0.0296, 'accuracy': 0.8058, 'neg_brier_score': -0.1485}

Random Forest (one hot encoding)
Year 1: {'roc_auc': 0.7174, 'f1': 0.2641, 'precision': 0.6389, 'recall': 0.1661, 'accuracy': 0.7227, 'neg_brier_score': -0.1844}
Year 2: {'roc_auc': 0.6838, 'f1': 0.0603, 'precision': 0.5521, 'recall': 0.0228, 'accuracy': 0.7462, 'neg_brier_score': -0.1767}
Year 3: {'roc_auc': 0.6528, 'f1': 0.0011, 'precision': 0.0, 'recall': 0.0011, 'accuracy': 0.8091, 'neg_brier_score': -0.1483}

XGBoost (one hot encoding)
Year 1: {'roc_auc': 0.7099, 'f1': 0.4334, 'precision': 0.5482, 'recall': 0.3599, 'accuracy': 0.7202, 'neg_brier_score': -0.1914}
Year 2: {'roc_auc': 0.6936, 'f1': 0.3144, 'precision': 0.4607, 'recall': 0.2401, 'accuracy': 0.7325, 'neg_brier_score': -0.184}
Year 3: {'roc_auc': 0.6406, 'f1': 0.1438, 'precision': 0.3235, 'recall': 0.0932, 'accuracy': 0.7894, 'neg_brier_score': -0.1606}

XGBoost (cat vars, LGBM)
Year 1: {'roc_auc': 0.6871, 'f1': 0.3222, 'precision': 0.5301, 'recall': 0.2341, 'accuracy': 0.7085, 'neg_brier_score': -0.1916}
Year 2: {'roc_auc': 0.7117, 'f1': 0.259, 'precision': 0.5167, 'recall': 0.1734, 'accuracy': 0.7462, 'neg_brier_score': -0.1723}
Year 3: {'roc_auc': 0.66, 'f1': 0.0816, 'precision': 0.3566, 'recall': 0.0463, 'accuracy': 0.802, 'neg_brier_score': -0.1501}

Multi-Level Perceptron (5,2) activation=relu
Year 1: {'roc_auc': 0.7191, 'f1': 0.4238, 'precision': 0.5689, 'recall': 0.3714, 'accuracy': 0.7229, 'neg_brier_score': -0.1893}
Year 2: {'roc_auc': 0.6656, 'f1': 0.2852, 'precision': 0.4336, 'recall': 0.1949, 'accuracy': 0.7375, 'neg_brier_score': -0.184}
Year 3: {'roc_auc': 0.5578, 'f1': 0.0639, 'precision': 0.271, 'recall': 0.0457, 'accuracy': 0.7919, 'neg_brier_score': -0.1512}

Multi-Level Perceptron (5,2) activation=tanh
Year 1: {'roc_auc': 0.7, 'f1': 0.4154, 'precision': 0.551, 'recall': 0.3606, 'accuracy': 0.7094, 'neg_brier_score': -0.1926}
Year 2: {'roc_auc': 0.6595, 'f1': 0.2205, 'precision': 0.3949, 'recall': 0.2131, 'accuracy': 0.7225, 'neg_brier_score': -0.1943}
Year 3: {'roc_auc': 0.5996, 'f1': 0.1615, 'precision': 0.2613, 'recall': 0.0782, 'accuracy': 0.7625, 'neg_brier_score': -0.1691}

Multi-Level Perceptron (50, 20) activation=relu
Year 1: {'roc_auc': 0.6204, 'f1': 0.4026, 'precision': 0.4088, 'recall': 0.4002, 'accuracy': 0.6467, 'neg_brier_score': -0.3375}
Year 2: {'roc_auc': 0.57, 'f1': 0.3274, 'precision': 0.371, 'recall': 0.3256, 'accuracy': 0.6741, 'neg_brier_score': -0.2742}
Year 3: {'roc_auc': 0.5444, 'f1': 0.2363, 'precision': 0.249, 'recall': 0.2356, 'accuracy': 0.7268, 'neg_brier_score': -0.2446}
'''
# Model choice
clf = DummyClassifier(strategy='stratified') # or 'major_class"
#clf = LogisticRegression(max_iter=1000)
#clf = RandomForestClassifier()
#clf = xgboost.XGBClassifier(objective='binary:logistic', use_label_encoder=False) # one hot encoding
#clf = lgb.LGBMClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1000, activation='relu')

# Dataset - dep. on Model Choice: df=dc.copy() if LGBMClassifier
df = d.copy()
#cols_X1, cols_X = cols_X1c, cols_Xc

# Performance measure choice
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']

################################################################################################
# get CV results

# Timer start
t0 = time()
# Model - Year 1
col_y = 'recidivism_arrest_year1'
y = df[col_y]
X = df[cols_X1]

dct_score = {}
for s in scores:
    fit_params = None
    if type(clf).__name__=='LGBMClassifier':
        fit_params = {'categorical_feature':cat_cols_1}
    dct_score[s] = round(cross_val_score(clf, X, y, cv=5, scoring=s, fit_params=fit_params).mean(), 4)
    print('CV score completed -- %s' % s)
print(dct_score)

# Model - Year 2, subset to those with no recidivism in Year 1
col_y = 'recidivism_arrest_year2'
y = df[df['recidivism_arrest_year1']==0][col_y]
X = df[df['recidivism_arrest_year1']==0][cols_X]
dct_score = {}
for s in scores:
    fit_params = None
    if type(clf).__name__=='LGBMClassifier':
        fit_params = {'categorical_feature':cat_cols}
    dct_score[s] = round(cross_val_score(clf, X, y, cv=5, scoring=s, fit_params=fit_params).mean(), 4)
print(dct_score)

# Model - Year 3, subset to those with no recidivism in Year 1 & 2
col_y = 'recidivism_arrest_year3'
y = df[(df['recidivism_arrest_year1']==0) & (df['recidivism_arrest_year2']==0)][col_y]
X = df[(df['recidivism_arrest_year1']==0) & (df['recidivism_arrest_year2']==0)][cols_X]
dct_score = {}
for s in scores:
    fit_params = None
    if type(clf).__name__=='LGBMClassifier':
        fit_params = {'categorical_feature':cat_cols}
    dct_score[s] = round(cross_val_score(clf, X, y, cv=5, scoring=s, fit_params=fit_params).mean(), 4)
print(dct_score)

# Timer ends
t1 = time()
print('Time elapsed for Year 123 CV results = {:.2f}'.format((t1-t0)))

