##################################
# NIJ Recidivism Challenge
# PCA implementation
# chris zhang 5/28/2021
##################################

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
import random
from time import time
import json
from aux_functions import *

## Read in data
d = read_and_clean_raw('./data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
# Recode all ordinal features with numerical-like values
cols_ordinal = ['age_at_release']
cols_ordinal += ['dependents', 'prison_years',
                 'prior_arrest_episodes_felony', 'prior_arrest_episodes_misd', 'prior_arrest_episodes_violent',
                 'prior_arrest_episodes_property', 'prior_arrest_episodes_drug',
                 'prior_arrest_episodes_ppviolationcharges', 'prior_conviction_episodes_felony',
                 'prior_conviction_episodes_misd', 'prior_conviction_episodes_prop', 'prior_conviction_episodes_drug']
cols_ordinal += ['delinquency_reports', 'program_attendances', 'program_unexcusedabsences', 'residence_changes']
for c in cols_ordinal:
    d[c] = get_ordinal_feature_col(d[c])

# Fill in NAs
print(d.isna().sum())
na_count = d.isna().sum()
na_cols = na_count[na_count>0].index
for c in na_cols:
    d[c] = fill_na(d[c])
# check dtype, set puma to str
#print(d.dtypes)
d['residence_puma'] = pd.Series(d['residence_puma'], dtype='str')

# Identify "Good Subgroups" that lead to good subgroup analysis results measured by Brier
'''
NN-based Good subgroups - see da_nij_subgroup.py output JSON

a.	prior_arrest_episodes_felony_1: "neg_brier_score": -0.1786
b.	prior_arrest_episodes_ppviolationcharges_0: "neg_brier_score": -0.1737
c.	prior_arrest_episodes_property_0: "neg_brier_score": -0.1778
d.	prior_conviction_episodes_felony_0: "neg_brier_score": -0.1845
e.	prior_conviction_episodes_misd_0: "neg_brier_score": -0.1851
f.	prior_conviction_episodes_prop_0: "neg_brier_score": -0.1837
g.	prior_conviction_episodes_ppviolationcharges_0: "neg_brier_score": -0.1866

NN-based Good subgroups PCA results NOT GOOD:
df = good_sub (top 3 NN Brier)
>>Best Brier Score = -0.1897513512760346, with 50 phenotypes
df = d
>>Best Brier Score = -0.18822946691927928, with 50 phenotypes
df = Year1/Never
>>Best Brier Score = -0.1921147317482997, with 35 phenotypes
df = good_sub (top 1 NN Brier), no ordinal, 122 features
>>Best Brier Score = -0.22031528004959475, with 95 phenotypes

'''
# dict from xvar to km cluster label represneting the "good subgroup"
# dct_sub = {'prior_arrest_episodes_felony': 1,
#            'prior_arrest_episodes_ppviolationcharges': 0,
#            'prior_arrest_episodes_property': 0,
#            'prior_conviction_episodes_felony': 0,
#            'prior_conviction_episodes_misd': 0,
#            'prior_conviction_episodes_prop': 0,
#            'prior_conviction_episodes_ppviolationcharges': 0
#            }
# dct_sub = {'prior_arrest_episodes_ppviolationcharges': 0,
#            }

# Identify PCA-based Good Subgroups:

# for each prior crime col, get clusters, run PCA for each cluster subgroup separately, and get performance
prior_crime_cols = get_prior_crime_cols()
# set a copy of d for sample restrictions over loop
d0 = d.copy()
# a dict to store PCA-subgroup output
dct_brier_pca_sub = {}
# PCA subgroup loop
for prior_crime_col in prior_crime_cols:
    for cluster_used in [0, 1]:
        print('Prior Crime Col = %s, cluster = %s' % (prior_crime_col, cluster_used))
        # name of col that stores km cluster labels
        km_col = 'km_%s' % prior_crime_col
        # get cluster labels for col
        d0[km_col] = get_km_subgroups(d0[prior_crime_col])
        # restrict d
        d = d0[d0[km_col] == cluster_used]
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^len(d) = %s' % len(d))
        if len(d)<500:
            pass
        else:
            # Set groupd id for naming output json file
            group_id = prior_crime_col + '_' + str(cluster_used)

            # One Hot Encoding
            # find boolean columns - set False, True = 0, 1 for bool cols
            bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
            for c in bool_cols:
                d[c] = [int(x) if not np.isnan(x) else np.nan for x in d[c]]
            print(two_class_cols) # manually encode two class cols
            d['female'] = np.where(d['gender']=='F', 1, 0)
            d['black'] = np.where(d['race']=='BLACK', 1, 0)
            # update binary_cols with female, black so excl. from cat_cols
            bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
            # cols not to be encoded
            cols_no_enc = ['id', 'supervision_risk_score_first',
                           'avg_days_per_drugtest', 'drugtests_thc_positive', 'drugtests_cocaine_positive',
                           'drugtests_meth_positive', 'drugtests_other_positive', 'percent_days_employed', 'jobs_per_year',]
            cols_no_enc += cols_ordinal
            # define categorica (3+ cats) cols
            cat_cols = set(d.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) - set(cols_no_enc)
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
            # for d (one hot encoding version)
            cols_X = [x for x in d.columns if x not in ['id']+ cols_ys]
            cols_X1 = [x for x in d.columns if (x not in ['id'] + cols_ys) and
                       (x.replace('_' + x.split('_')[-1], '') not in cols_sup_act) and
                       (x not in cols_sup_act)] # remove sup act cols for year 1 features

            # Fit PCA
            # set master data table
            df = d.copy()
            # idxs_year1_never = d[(d['recidivism_arrest_year1']==1) | (d['recidivism_within_3years']==0)].index
            # df = d.loc[idxs_year1_never, ].copy()

            # model and evaluation score
            clf = LogisticRegression(max_iter=5000)
            scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
            # training data
            col_y = 'recidivism_arrest_year1'
            y = df[col_y]
            X_pca = df[cols_X1]
            # optimize wrt opt_n
            n_comps = list(range(5, X_pca.shape[1]+5, 5))
            best_brier = -999
            opt_n = -999
            for n_comp in n_comps:
                n_comp = min(n_comp, min(X_pca.shape[1], X_pca.shape[0])) # rare events may also be short (a few rows)
                print('************\nPCA: n_components = %s' % n_comp)
                X_reduced = PCA(n_components=n_comp).fit_transform(X_pca)
                dct_score = cross_validate(clf, X_reduced, y, cv=5, scoring=scores)
                print('Average Score:', {k: round(v.mean(), 6) for k, v in dct_score.items()})
                brier = dct_score['test_neg_brier_score'].mean()
                if brier > best_brier:
                    best_brier=brier
                    opt_n = n_comp

            dct_brier_pca_sub['%s_%s' % (prior_crime_col, cluster_used)] = (round(best_brier, 6), opt_n)
            print('*****[ DONE ]*****Prior Crime Col = %s, cluster = %s' % (prior_crime_col, cluster_used))
            print('>>Best Brier Score = %s, with %s phenotypes' % (best_brier, opt_n))

# Export dct_brier_pca_sub
fp_out = './output/dct_brier_pca_sub.json'
with open(fp_out, 'w') as f:
    json.dump(dct_brier_pca_sub, f)

# Get and see PCA-based Good Subgroups:
from collections import OrderedDict
thre_good_sub = 0.18 # max abs brier for flagging good subgroup (0.1883=max score PCA baseline)
dct_good_pca_sub = OrderedDict({k: v for k, v in dct_brier_pca_sub.items() if abs(v[0])<=thre_good_sub})
dct_good_pca_sub = OrderedDict(sorted(dct_good_pca_sub.items(), key=lambda z: abs(z[1][0])))
for k, v in dct_good_pca_sub.items():
    print(k, v)

'''
prior_arrest_episodes_ppviolationcharges_1 (-0.164126, 30)
prior_arrest_episodes_misd_1 (-0.167449, 40)
prior_arrest_episodes_misd_0 (-0.167457, 40)
prior_conviction_episodes_prop_0 (-0.173532, 35)
prior_conviction_episodes_prop_1 (-0.173624, 35)
prior_conviction_episodes_felony_1 (-0.175658, 35)
prior_conviction_episodes_ppviolationcharges_1 (-0.178448, 40)
prior_arrest_episodes_felony_0 (-0.181519, 35)
prior_arrest_episodes_violent_0 (-0.184759, 50)
prior_arrest_episodes_guncharges_0 (-0.186217, 35)
prior_conviction_episodes_domesticviolencecharges_0 (-0.186757, 40)
prior_conviction_episodes_guncharges_0 (-0.187403, 35)

'''
# Restore d as d0
# Note that d0 contains km_ cols
d = d0.copy()

# Make PCA-based good_sub df
# init empty dataframe to store good subgroups
good_sub = pd.DataFrame()
# add in new rows identified by good subgroups
for k, v in dct_good_pca_sub.items():
    # get col label and numerical cluster label
    cluster = k.split('_')[-1]
    col = k.replace('_%s' % cluster, '') # e.g. prior_arrest_episodes_felony
    km_col = 'km_' + col # e.g. km_e.g. prior_arrest_episodes_felony
    cluster = int(cluster)
    # get new indices identified by good subgroup
    n_good_sub = len(set(d.loc[d[km_col]==cluster, ].index))
    print('******\nCol = %s, cluster = %s, n_comp=%s' % (col, cluster, v[1]))
    print('Total size of good subgroup = %s' % n_good_sub)
    idxs_new_rows = set(d.loc[d[km_col]==cluster, ].index) - set(good_sub.index)
    print('Size of additional rows identified as good sub = %s' % len(idxs_new_rows))
    # append new rows to good_sub
    good_sub = good_sub.append(d.loc[idxs_new_rows, ])
    print('Total size of good subgroup accumulated so far = %s' % len(good_sub))
    del good_sub[km_col]

'''
******
Col = prior_arrest_episodes_ppviolationcharges, cluster = 1, n_comp=30
Total size of good subgroup = 10079
Size of additional rows identified as good sub = 10079
Total size of good subgroup accumulated so far = 10079
******
Col = prior_arrest_episodes_misd, cluster = 1, n_comp=40
Total size of good subgroup = 9282
Size of additional rows identified as good sub = 1823
Total size of good subgroup accumulated so far = 11902
******
Col = prior_arrest_episodes_misd, cluster = 0, n_comp=40
Total size of good subgroup = 8746
Size of additional rows identified as good sub = 6126
Total size of good subgroup accumulated so far = 18028
******
Col = prior_conviction_episodes_prop, cluster = 0, n_comp=35
Total size of good subgroup = 6121
Size of additional rows identified as good sub = 0
Total size of good subgroup accumulated so far = 18028
'''

# Final Model candidate:
# Fit optimal PCA on i-th best subgroup, predict on corresponding subgroup in test data that has not been predicted

# Evaluate the Final Model
from sklearn.model_selection import train_test_split
#############################
# One Hot Encoding d = d0.copy()
#############################
#  One Hot Encoding
# find boolean columns - set False, True = 0, 1 for bool cols
bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
for c in bool_cols:
    d[c] = [int(x) if not np.isnan(x) else np.nan for x in d[c]]
print(two_class_cols)  # manually encode two class cols
d['female'] = np.where(d['gender'] == 'F', 1, 0)
d['black'] = np.where(d['race'] == 'BLACK', 1, 0)
# update binary_cols with female, black so excl. from cat_cols
bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(d)
# cols not to be encoded
cols_no_enc = ['id', 'supervision_risk_score_first',
               'avg_days_per_drugtest', 'drugtests_thc_positive', 'drugtests_cocaine_positive',
               'drugtests_meth_positive', 'drugtests_other_positive', 'percent_days_employed', 'jobs_per_year', ]
cols_no_enc += cols_ordinal
# define categorica (3+ cats) cols
cat_cols = set(d.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) - set(cols_no_enc)
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
# for d (one hot encoding version)
cols_X = [x for x in d.columns if x not in ['id'] + cols_ys]
cols_X1 = [x for x in d.columns if (x not in ['id'] + cols_ys) and
           (x.replace('_' + x.split('_')[-1], '') not in cols_sup_act) and
           (x not in cols_sup_act)]  # remove sup act cols for year 1 features
# Model training
# PCA on 1st best subgroup
train_d,valid_d = train_test_split(d, test_size=0.2,random_state=13)
X_pca_1 = train_d[train_d['km_prior_arrest_episodes_ppviolationcharges']==1][cols_X1]
X_reduced_1 = PCA(n_components=30).fit_transform(X_pca_1)
y_1 = train_d['recidivism_arrest_year1'][X_pca_1.index]

Xv_pca_1 = valid_d[valid_d['km_prior_arrest_episodes_ppviolationcharges']==1][cols_X1]
Xv_reduced_1 = PCA(n_components=30).fit_transform(Xv_pca_1)
yv_1 = valid_d['recidivism_arrest_year1'][Xv_pca_1.index]

clf = LogisticRegression(max_iter=5000)
clf_1 = clf.fit(X_reduced_1, y_1)
phat_1 = clf_1.predict_proba(Xv_reduced_1)
phat_1 = phat_1[:,1] # prob(=1)
yhat_1 = clf_1.predict(Xv_reduced_1)

valid_d['phat'] = np.nan
valid_d['yhat'] = np.nan
valid_d.loc[Xv_pca_1.index, 'phat'] = phat_1
valid_d.loc[Xv_pca_1.index, 'yhat'] = yhat_1

# PCA on 2nd best subgroup - train using all subgroup 2, validate using ADDITIONAL rows in subgroup 2
# Or train using entire train_d with n_components=35 (the ADDITIONAL rows are hard to be predited by a subgroup-2 PCA model!)
#X_pca_2 = train_d[train_d['km_prior_arrest_episodes_misd']==1][cols_X1] # use n_components=40
X_pca_2 = train_d[cols_X1]
X_reduced_2 = PCA(n_components=35).fit_transform(X_pca_2)
y_2 = train_d['recidivism_arrest_year1'][X_pca_2.index]

Xv_pca_2 = valid_d[(valid_d['km_prior_arrest_episodes_ppviolationcharges']==0) &
                  (valid_d['km_prior_arrest_episodes_misd']==1)][cols_X1]
Xv_reduced_2 = PCA(n_components=35).fit_transform(Xv_pca_2)
yv_2 = valid_d['recidivism_arrest_year1'][Xv_pca_2.index]

clf = LogisticRegression(max_iter=5000)
clf_2 = clf.fit(X_reduced_2, y_2)
phat_2 = clf_2.predict_proba(Xv_reduced_2)
phat_2 = phat_2[:,1] # prob(=1)
yhat_2 = clf_2.predict(Xv_reduced_2)

valid_d.loc[Xv_pca_2.index, 'phat'] = phat_2
valid_d.loc[Xv_pca_2.index, 'yhat'] = yhat_2

# PCA on 3rd best subgroup - train using all subgroup 3, validate using ADDITIONAL rows in subgroup 3
# Or train using entire train_d with n_components=35 (the ADDITIONAL rows are hard to be predited by a subgroup-3 PCA model!)
#X_pca_3 = train_d[train_d['prior_arrest_episodes_misd']==0][cols_X1] # use n_components=40
X_pca_3 = train_d[cols_X1]
X_reduced_3 = PCA(n_components=35).fit_transform(X_pca_3)
y_3 = train_d['recidivism_arrest_year1'][X_pca_3.index]

Xv_pca_3 = valid_d[(valid_d['km_prior_arrest_episodes_ppviolationcharges']==0) &
                  (valid_d['km_prior_arrest_episodes_misd']==0) &
                  (valid_d['km_prior_arrest_episodes_misd']==0)][cols_X1]
Xv_reduced_3 = PCA(n_components=35).fit_transform(Xv_pca_3)
yv_3 = valid_d['recidivism_arrest_year1'][Xv_pca_3.index]

clf = LogisticRegression(max_iter=5000)
clf_3 = clf.fit(X_reduced_3, y_3)
phat_3 = clf_3.predict_proba(Xv_reduced_3)
phat_3 = phat_3[:,1] # prob(=1)
yhat_3 = clf_3.predict(Xv_reduced_3)

valid_d.loc[Xv_pca_3.index, 'phat'] = phat_3
valid_d.loc[Xv_pca_3.index, 'yhat'] = yhat_3

# performance scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
y_true = valid_d['recidivism_arrest_year1']
yhat = valid_d['yhat']
phat = valid_d['phat']

def get_scores(y_true, yhat, phat):
    scores = {}
    scores['roc_auc'] = roc_auc_score(y_true, phat)
    scores['accuracy'] = accuracy_score(y_true, yhat)
    scores['precision'] = precision_score(y_true, yhat)
    scores['recall'] = recall_score(y_true, yhat)
    scores['f1'] = f1_score(y_true, yhat)
    scores['brier'] = brier_score_loss(y_true, phat)
    scores = {k:round(v, 6) for k, v in scores.items()}
    print(scores)
    return None

get_scores(yv_1, yhat_1, phat_1)
get_scores(yv_2, yhat_2, phat_2)
get_scores(yv_3, yhat_3, phat_3)
get_scores(y_true, yhat, phat)

############
# Final Forecast
############
dt = read_and_clean_raw('./data/nij/NIJ_s_Recidivism_Challenge_Test_Dataset1.csv')
# Recode all ordinal features with numerical-like values
cols_ordinal = ['age_at_release']
cols_ordinal += ['dependents', 'prison_years',
                 'prior_arrest_episodes_felony', 'prior_arrest_episodes_misd', 'prior_arrest_episodes_violent',
                 'prior_arrest_episodes_property', 'prior_arrest_episodes_drug',
                 'prior_arrest_episodes_ppviolationcharges', 'prior_conviction_episodes_felony',
                 'prior_conviction_episodes_misd', 'prior_conviction_episodes_prop', 'prior_conviction_episodes_drug']
cols_ordinal += ['delinquency_reports', 'program_attendances', 'program_unexcusedabsences', 'residence_changes']
for c in cols_ordinal:
    dt[c] = get_ordinal_feature_col(d[c])

# Fill in NAs
print(dt.isna().sum())
na_count = dt.isna().sum()
na_cols = na_count[na_count>0].index
for c in na_cols:
    dt[c] = fill_na(dt[c])
# check dtype, set puma to str
#print(d.dtypes)
dt['residence_puma'] = pd.Series(dt['residence_puma'], dtype='str')

###############
# One Hot Encoding for dt
###############
#  One Hot Encoding
# find boolean columns - set False, True = 0, 1 for bool cols
bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(dt)
for c in bool_cols:
    dt[c] = [int(x) if not np.isnan(x) else np.nan for x in dt[c]]
print(two_class_cols)  # manually encode two class cols
dt['female'] = np.where(dt['gender'] == 'F', 1, 0)
dt['black'] = np.where(dt['race'] == 'BLACK', 1, 0)
# update binary_cols with female, black so excl. from cat_cols
bool_cols, binary_cols, two_class_cols = get_bool_binary_cols(dt)
# cols not to be encoded
cols_no_enc = ['id', 'supervision_risk_score_first',
               'avg_days_per_drugtest', 'drugtests_thc_positive', 'drugtests_cocaine_positive',
               'drugtests_meth_positive', 'drugtests_other_positive', 'percent_days_employed', 'jobs_per_year', ]
cols_no_enc += cols_ordinal
# define categorica (3+ cats) cols
cat_cols = set(dt.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) - set(cols_no_enc)
# for master dt, do one hot encoding for cat cols (for sk-learn models)
# one hot encoding
# Note 1: set drop_first=False for purely linear logit (statsmodel)
# Note 2: cols with diff dtype will not be encoded, so puma must be converted to str first
dummies = pd.get_dummies(dt[cat_cols], drop_first=False)
dt = dt.join(dummies)
dt = dt.drop(columns=list(cat_cols) + two_class_cols)

# Identify subgroups in dt using km-thresholds in training data
km_subs = [('prior_arrest_episodes_ppviolationcharges', 1),
               ('prior_arrest_episodes_misd', 1),
               ('prior_arrest_episodes_misd', 0)]
for col, cluster in km_subs:
    print('************\nCol = %s, Cluster = %s' % (col, cluster))
    print(d[d['km_%s' % col]==cluster][col].value_counts().sort_index())
dt['km_prior_arrest_episodes_ppviolationcharges'] = np.where(dt['prior_arrest_episodes_ppviolationcharges']
                                                             .isin([0, 1, 2]), 1, 0)
dt['km_prior_arrest_episodes_misd'] = np.where(dt['prior_arrest_episodes_misd']
                                                             .isin([0, 1, 2, 3]), 1, 0)

'''
Col = prior_arrest_episodes_ppviolationcharges, cluster = 1, n_comp=30
******
Col = prior_arrest_episodes_misd, cluster = 1, n_comp=40 (use 35 for full sample PCA)
******
Col = prior_arrest_episodes_misd, cluster = 0, n_comp=40 (use 35 for full sample PCA)
'''

# Model training and forecasting
# re-confirm cols_X1
# dt does not have sup_act cols as of Year 1 data release, just exclude id, km_ cols
cols_X1 = [x for x in dt.columns if x not in['id', 'km_prior_arrest_episodes_ppviolationcharges',
                                             'km_prior_arrest_episodes_misd']]
# init phat in dt
dt['phat']=np.nan
# PCA on 1st best subgroup
X_pca_1 = d[d['km_prior_arrest_episodes_ppviolationcharges']==1][cols_X1]
X_reduced_1 = PCA(n_components=30).fit_transform(X_pca_1)
y_1 = d['recidivism_arrest_year1'][X_pca_1.index]

clf = LogisticRegression(max_iter=5000)
clf_1 = clf.fit(X_reduced_1, y_1)
Xt_pca_1 = dt[dt['km_prior_arrest_episodes_ppviolationcharges']==1][cols_X1]
Xt_reduced_1 = PCA(n_components=30).fit_transform(Xt_pca_1)
phat_1 = clf_1.predict_proba(Xt_reduced_1)
phat_1 = phat_1[:,1]
dt.loc[Xt_pca_1.index, 'phat'] = phat_1

# PCA on 2nd best subgroup - train using all subgroup 2, validate using ADDITIONAL rows in subgroup 2
# Or train using entire train_d with n_components=35 (the ADDITIONAL rows are hard to be predited by a subgroup-2 PCA model!)
#X_pca_2 = train_d[train_d['km_prior_arrest_episodes_misd']==1][cols_X1] # use n_components=40
X_pca_2 = d[cols_X1]
X_reduced_2 = PCA(n_components=35).fit_transform(X_pca_2)
y_2 = d['recidivism_arrest_year1'][X_pca_2.index]

clf = LogisticRegression(max_iter=5000)
clf_2 = clf.fit(X_reduced_2, y_2)
Xt_pca_2 = dt[(dt['km_prior_arrest_episodes_ppviolationcharges']==0) &
                  (dt['km_prior_arrest_episodes_misd']==1)][cols_X1]
Xt_reduced_2 = PCA(n_components=35).fit_transform(Xt_pca_2)
phat_2 = clf_2.predict_proba(Xt_reduced_2)
phat_2 = phat_2[:,1]
dt.loc[Xt_pca_2.index, 'phat'] = phat_2

# PCA on 3rd best subgroup - train using all subgroup 3, validate using ADDITIONAL rows in subgroup 3
# Or train using entire train_d with n_components=35 (the ADDITIONAL rows are hard to be predited by a subgroup-3 PCA model!)
#X_pca_3 = train_d[train_d['prior_arrest_episodes_misd']==0][cols_X1] # use n_components=40
X_pca_3 = d[cols_X1]
X_reduced_3 = PCA(n_components=35).fit_transform(X_pca_3)
y_3 = d['recidivism_arrest_year1'][X_pca_3.index]

clf = LogisticRegression(max_iter=5000)
clf_3 = clf.fit(X_reduced_3, y_3)
Xt_pca_3 = dt[(dt['km_prior_arrest_episodes_ppviolationcharges']==0) &
                  (dt['km_prior_arrest_episodes_misd']==0) &
                  (dt['km_prior_arrest_episodes_misd']==0)][cols_X1]
Xt_reduced_3 = PCA(n_components=35).fit_transform(Xt_pca_3)
phat_3 = clf_3.predict_proba(Xt_reduced_3)
phat_3 = phat_3[:,1]
dt.loc[Xt_pca_3.index, 'phat'] = phat_3

# Export as submission file
submit_year1 = dt[['id', 'phat']].sort_values(by='id', ascending=True)
submit_year1['phat'] = [round(x, 4) for x in submit_year1['phat']]
submit_year1 = submit_year1.rename(columns={'id':'ID', 'phat':'Probability'})
submit_year1.to_excel('./output/American Institutes for Research_1YearForecast.xlsx', index=False)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.kdeplot(data=submit_year1['Probability'])
plt.show()


















