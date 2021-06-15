##################################
# NIJ Recidivism Challenge
# Year 2 model - based on subgroup analysis
# chris zhang 6/14/2021
##################################

# TODO [done]: ues categorical cols for XGBoost
# TODO: for minor class pick best thre for prediction, based on maximizing alt metrics (e.g. G-mean, Youden)
# https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
# TODO [done]: baseline: random, one class, L2 logit
# TODO [done]: neural network
# TODO [done]: add in geo chars
# TODO [done - not improving]: add path xvars with dict for relevant risky prior crimes (either arrest 3+, conv 1+)

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

# Read in data
d = read_and_clean_raw('./data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')

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
# Check prevalence of path forming by current prison_offense
for k, v in dct_reci_path.items():
    print(pd.crosstab(d['prison_offense_%s' % k], d['path_%s' % k]))
    print('-'*50)

# Define cols for model training's X, y
cols_ys = ['recidivism_within_3years', 'recidivism_arrest_year1', 'recidivism_arrest_year2', 'recidivism_arrest_year3']
# supervision activity cols, to be excl. from year 1 model
cols_sup_act = get_sup_act_cols()
# for d (one hot encoding version)
cols_X = [x for x in d.columns if x not in ['id']+ cols_ys]
cols_X1 = [x for x in d.columns if (x not in ['id'] + cols_ys) and
           (x.replace('_' + x.split('_')[-1], '') not in cols_sup_act) and
           (x not in cols_sup_act)] # remove sup act cols for year 1 features

# for dc (cat col version), convert cat cols to integers so readable by lgb
# exceptions: puma (keep orig codes)
# TODO: keep a mapping between cats and codes

for c in [x for x in cat_cols if x not in ['residence_puma']]:
    dc[c] = pd.Categorical(dc[c]).codes
cols_Xc = [x for x in dc.columns if x not in ['id']+ cols_ys]
cols_X1c = [x for x in dc.columns if x not in ['id'] + cols_ys + cols_sup_act] # remove sup act cols for year 1 features
# cat_cols include the 16 supervision cols, remove for Year 1
cat_cols_1 = [x for x in cat_cols if x not in cols_sup_act]

####################################################################################
# Year 2 Model
# 1. Find optimal PCA/Logit setting for path_xvars subgroups
# 2. For best-performing subgroups, use corresponding PCA/Logit
# 3. For other subgroups, use a single neural network model
####################################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# clf = DummyClassifier(strategy='stratified') # or 'major_class"
clf = LogisticRegression(max_iter=5000)
#clf = RandomForestClassifier()
# clf = xgboost.XGBClassifier(objective='binary:logistic', use_label_encoder=False) # one hot encoding
#clf = lgb.LGBMClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=1000, activation='relu')

# Dataset - dep. on Model Choice: df=dc.copy() if LGBMClassifier
df = d.copy()
# master dct for subgroup results
dct_subgroup_master = {}
# subgrouping
path_xvars = [c for c in d.columns if 'path_' in c]
for path_xvar in path_xvars:
    for subgroup_bool in [True, False]:
        print('path_xvars = %s. Condition = %s' % (path_xvar, subgroup_bool))
        df = d[(d[path_xvar]>0)==subgroup_bool]

        # Set groupd id for naming output json file
        group_id = path_xvar + '_' + str(int(subgroup_bool))

        # Performance measure choice
        scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']

        ################################################################################################
        # get CV results

        # Timer start
        t0 = time()
        # Data
        col_y = 'recidivism_arrest_year2'
        y = df[df['recidivism_arrest_year1']==0][col_y]
        X = df[df['recidivism_arrest_year1']==0][cols_X]

        pipe = Pipeline([("scale", StandardScaler()),
                         ("reduce_dims", PCA()),
                         ('model', clf)])
        param_grid = dict(reduce_dims__n_components=[0.25, 0.5, 0.75],
                          )
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring=scores, refit='neg_brier_score')
        grid.fit(X, y)

        print('Best Params: ', grid.best_params_)
        print('Best Scores: ', grid.best_score_)
        t1 = time()
        print('Time elapse = %s for finishing PCA training for path_xvar, bool = (%s, %s)' %
              (int(t1-t0), path_xvar, subgroup_bool))
        print('-'*150)
        # assign to dct_subgroup_master
        dct_subgroup_master[group_id] = {'brier': grid.best_score_, 'opt_n': grid.best_params_}
        #
        # # Standardize data for PCA
        # scaler = preprocessing.StandardScaler().fit(X)
        # X_std = scaler.transform(X)
        #
        # # PCA - Year 2
        # X_pca = np.array(X)
        # n_comps = list(range(5, X_pca.shape[1]+5, 5))
        # best_brier = -999
        # opt_n = -999
        # for n_comp in n_comps:
        #     n_comp = min(n_comp, X_pca.shape[1])
        #     print('************\nPCA: n_components = %s' % n_comp)
        #     X_reduced = PCA(n_components=n_comp).fit_transform(X)
        #     dct_score = cross_validate(clf, X_reduced, y, cv=5, scoring=scores)
        #     print('Average Score:', {k: round(v.mean(), 6) for k, v in dct_score.items()})
        #     brier = dct_score['test_neg_brier_score'].mean()
        #     if brier > best_brier:
        #         best_brier=brier
        #         opt_n = n_comp
        # print('>>Best Brier Score = %s, with %s components' % (best_brier, opt_n))
        #
        # # assign to dct_subgroup_master
        # dct_subgroup_master[group_id] = {'brier': best_brier, 'opt_n': opt_n}

# Export scores
dct_subgroup_master = OrderedDict(sorted(dct_subgroup_master.items(), key=lambda x: x[1]['brier'],
                                             reverse=True))
fp_out = './output/year_2_forecast/dct_subgroup_master.json'  # group_id defined when subgrouping d
with open(fp_out, 'w') as f:
    json.dump(dct_subgroup_master, f, indent=4)

# get subgrouping
df = d.copy()
df = df[df['recidivism_arrest_year1']==0]

print(df[df['path_Violent/Sex']>0].shape)

print(df[(df['path_Violent/Sex']==0) &
         (df['path_Property']==0)].shape)

print(df[(df['path_Violent/Sex']==0) &
         (df['path_Property']>0) &
         (df['path_Violent/Non-Sex'] > 0)].shape)

print(df[(df['path_Violent/Sex']==0) &
         (df['path_Property']>0) &
         (df['path_Violent/Non-Sex'] == 0) &
         (df['path_Drug']>0)].shape)

print(df[(df['path_Violent/Sex']==0) &
         (df['path_Property']>0) &
         (df['path_Violent/Non-Sex'] == 0) &
         (df['path_Drug'] == 0)].shape)
'''
# Brier of top subgroups
    "path_Violent/Sex_1": {
        "brier": -0.12248489263482809,
        "opt_n": {
            "reduce_dims__n_components": 0.25
        }
    },
    "path_Property_0": {
        "brier": -0.17218054226725305,
        "opt_n": {
            "reduce_dims__n_components": 0.75
        }
    },
    "path_Violent/Non-Sex_1": {
        "brier": -0.17234179137930777,
        "opt_n": {
            "reduce_dims__n_components": 0.5
        }
    },
    "path_Drug_1": {
        "brier": -0.17385753975833346,
        "opt_n": {
            "reduce_dims__n_components": 0.5
        }
    },
'''

# Hard-to-predict subgroups - Year 2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.core import Lambda
from keras import backend as K
import matplotlib.pyplot as plt

# Data data set up
cond_rest = ((df['path_Violent/Sex']==0) &
         (df['path_Property']>0) &
         (df['path_Violent/Non-Sex'] == 0) &
         (df['path_Drug'] == 0))
col_y = 'recidivism_arrest_year2'
y = df[cond_rest][col_y]
X = df[cond_rest][cols_X]

################################################
# Hard-to-predict subgroups
# ML methods - RF performs best over logit, Ada, XGB
# RF n_estimators = 400 works fine (brier = 0.191+)
################################################
# clfs = [LogisticRegression(max_iter=5000)]
# clfs.append(RandomForestClassifier())
# clfs.append(AdaBoostClassifier())
# clfs.append(xgboost.XGBClassifier(objective='binary:logistic', use_label_encoder=False))
clfs = [RandomForestClassifier(n_estimators=x) for x in range(100, 2000, 100)]
for clf in clfs:
    z = cross_validate(clf, X, y, cv=5, scoring=scores)
    #print('clf = %s.' % (clf.__class__.__name__), 'Brier = {brier: .6f}'.format(brier=z['test_neg_brier_score'].mean()))
    print('n_estimator = %s. clf = %s.' % (clf.n_estimators, clf.__class__.__name__), 'Brier = {brier: .6f}'.format(brier=z['test_neg_brier_score'].mean()))

################################################
# Hard-to-predict subgroups
# Neural Network - cannot outperform ML methods
################################################
train_X, valid_X, train_y, valid_y= train_test_split(X, y, test_size=0.2)
# scaler = preprocessing.StandardScaler().fit(train_X)
# train_X = scaler.transform(train_X)
# scaler = preprocessing.StandardScaler().fit(valid_X)
# valid_X = scaler.transform(valid_X)
# NN - model set up
def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))
model = Sequential()
model.add(Dense(32, activation='relu', name='layer1'))
model.add(Dense(64, activation='relu', name='layer2'))
model.add(Dropout(0.2))
# model.add(PermaDropout(0.2)),
model.add(Dense(64, activation='tanh', name='layer3'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])
epochs = 300
batch_size = 1000
model.fit(train_X, train_y, epochs=epochs, batch_size=500, verbose=1, validation_data=(valid_X, valid_y))

# Training vs Validation Loss Plot
model_history = model.history.__dict__['history']
loss = model_history['loss']
val_loss = model_history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#######################################


