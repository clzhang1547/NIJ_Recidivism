##################################
# NIJ Recidivism Challenge
# Denoise Autoencoder - Implementation with NIJ data
# chris zhang 5/18/2021
#
# TODO [done]: check DA loss graph on denoising x_train_noisy <- noise factor=0.01 works for NIJ data
# ^noise factor=0.02 divegence occurs, 0.1, 0.5 bad
# TODO [done]: add constant xvars so #cols = 4X, 8X allowing encoding/decoding flexibility to lower dim
# TODO: [done] noise=0.5 good, pred on d (no noise). Try noise=1, pred on d, until noise too large (hurting perf)
#^cannot overwrite autoencoder obj with new noise_factor ow diverging, rerun code to ensure valid result
# TODO: [done] set noise a la Beaulieu-Jones 2016: randomly set to 0 for 20% values of input matrix
# TODO: [done] Year 1 results with 128>>64 xvars and 0.5 noise get similar/worse perf vs ML. Try 32 or 8 xvars, random 0 as noise
# TODO: [done - not applicable?] check RNN for DA
# TODO: [done] try dropping NA rows (fillna may introduce noise) - not improving
# TODO: [done] check subgroups (arrest/conviction types) - not improving
# TODO: no DA, use orig d for autoencoder and get phenotypes, run XGBoost/Shapley to get top phenotypes, run ML
# TODO: add in ACS/GA crime xvars (re Mason/Sandeep)
# TODO: optimize wrt phat thre for 0/1 to improve performance, check FP/FN along distribution of phat
# TODO: check the combination of different denoised features (phenotypes) for prediction

##################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv1D,MaxPooling1D,UpSampling1D
from keras.models import Model

from sklearn.model_selection import train_test_split

import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
from aux_functions import *

from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json

prior_crime_cols = get_prior_crime_cols()
layer_counter = 0 # upsampling layer name suffix, +2 in each loop of prior_crime_col
for prior_crime_col in prior_crime_cols:
    print('='*50)
    print('KM-subgroup analysis - c = %s' % prior_crime_col)
    print('=' * 50)
    # Read in data
    # d will be purely train data (no id, no yvars)
    d = read_and_clean_raw(fp='data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
    cols_ys = ['recidivism_within_3years', 'recidivism_arrest_year1', 'recidivism_arrest_year2', 'recidivism_arrest_year3']
    yvar = 'recidivism_arrest_year1'
    train_labels = d[yvar].astype(int)
    d = d[[x for x in d.columns if x not in cols_ys]]
    # get supervision activity cols
    cols_sup_act = get_sup_act_cols()

    # Fill in NAs
    print(d.isna().sum())
    na_count = d.isna().sum()
    na_cols = na_count[na_count>0].index
    # NA option 1 - fill with random valid values
    for c in na_cols:
        d[c] = fill_na(d[c])
    # NA option 2 - drop rows with any NA in relevant xvars
    # if yvar == 'recidivism_arrest_year1':
    #     d = d.dropna(how='any', subset=set(na_cols) - set(cols_sup_act))
    # if yvar in ['recidivism_arrest_year2', 'recidivism_arrest_year3']:
    #     d = d.dropna(how='any', subset=set(na_cols))
    # check dtype, set puma to str
    #print(d.dtypes)
    d['residence_puma'] = pd.Series(d['residence_puma'], dtype='str')
    # Subgrouping

    # subgroup option 1 - using labels
    # d = d[d['supervision_risk_score_first']>=6]
    # d = d[d['age_at_release'].isin(['23-27', '28-32', '33-37'])]
    # d = d[d['gang_affiliated']==1]
    # d = d[~d['prior_arrest_episodes_ppviolationcharges'].isin(['0'])]

    # subgroup option 2  - K means clustering (for Prior GA Criminal History xvars)
    xvar_subgroup = prior_crime_col
    d['km_%s' % xvar_subgroup] = get_km_subgroups(d[xvar_subgroup])
    cluster_used = 1
    d = d[d['km_%s' % xvar_subgroup]==cluster_used]

    # Set groupd id for naming output json file
    group_id = xvar_subgroup + '_' + str(cluster_used)
    # update train_labels with non-NA index in d
    train_labels = train_labels[d.index]

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
    # supervision activity cols, to be excl. from year 1 model
    if yvar == 'recidivism_arrest_year1':
        d = d.drop(columns=cols_sup_act)
    # define categorical (3+ cats) cols
    cat_cols = set(d.columns) - set(bool_cols) - set(binary_cols) - set(two_class_cols) \
               - set(cols_no_enc) - set(['female', 'black', 'id'])

    # add dummies to d
    dummies = pd.get_dummies(d[cat_cols], drop_first=False)
    d = d.join(dummies)
    d = d.drop(columns=list(cat_cols) + two_class_cols)
    del d['id']

    # drop some xvars to ensure # vars = 4x for 2 Maxpooling1D with size=2
    # del d['prior_arrest_episodes_property_1']
    # del d['prior_arrest_episodes_property_2']

    # add constant xvars to ensure # vars = 4x for 2 Maxpooling1D with size=2
    for i in range(128 - d.shape[1]):
        d['z_%s' % i] = 1

    # Reshape train data
    d = np.array(d)
    n, k = d.shape
    d = d.reshape(-1, k, 1)

    # Rescale train data
    d = d/np.max(d)

    # split train for validation
    train_X,valid_X,train_ground,valid_ground = train_test_split(d,
                                                                 d,
                                                                 test_size=0.2,
                                                                 random_state=13)

    # Autoencoder
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

    # Add noise to train/valid data
    # noise_factor = 0.5 # 0.5 on NIJ data would cause valid loss diverging up, 0.01 is good for convergence
    # x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
    # x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)
    # x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    # x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)
    noise_prop = 0.2
    x_train_noisy = set_random_nonzero_elements(train_X, noise_prop, fill=0)
    x_valid_noisy = set_random_nonzero_elements(valid_X, noise_prop, fill=0)

    # Denoise Autoencoder Hyperparameters
    batch_size = 128
    epochs = 20
    input_img = Input(shape = (k, 1))

    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
    # Train model - DA (note x_train_noisy and train_X are same shape)
    autoencoder_train = autoencoder.fit(x_train_noisy, train_X, batch_size=batch_size,epochs=epochs,
                                        verbose=1,validation_data=(x_valid_noisy, valid_X))

    # Training vs Validation Loss Plot
    # loss = autoencoder_train.history['loss']
    # val_loss = autoencoder_train.history['val_loss']
    # epochs = range(epochs)
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.show()

    # Get intermediate layer (phenotype) from trained model
    # get second last UpSampling1D layer
    if layer_counter==0:
        layer_name = 'up_sampling1d'
    elif layer_counter>0:
        layer_name = 'up_sampling1d' + '_%s' % layer_counter

    intermediate_layer_model = Model(inputs=autoencoder.input,
                                     outputs=autoencoder.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(d) # N x k/4 x 128, get phenotype for entire dataset d
    phenotype = Conv1D(1, (3,), activation='sigmoid', padding='same')(intermediate_output)  # N x k/4 x 1
    phenotype = phenotype.numpy()[:,:,0]
    # XGBoost for phenotype and train_labels
    # clf = xgboost.XGBClassifier(objective='binary:logistic', use_label_encoder=False) # one hot encoding
    clf = LogisticRegression(max_iter=1000)
    # clf = RandomForestClassifier()
    scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
    # Model - Year 1
    y = train_labels
    X = phenotype
    dct_score = {}
    for s in scores:
        v = round(cross_val_score(clf, X, y, cv=5, scoring=s).mean(), 4)
        dct_score[s] = v
        print('CV score completed -- %s=%s' % (s, v))
    print(dct_score)

    # Export scores
    fp_out = './output/score_%s_%s.json' % (yvar.split('_')[-1], group_id) # group_id defined when subgrouping d
    with open(fp_out, 'w') as f:
        json.dump(dct_score, f)

    # del autoencoder from memory so up_sampling1d layer name can be recycled
    # del autoencoder
    # del autoencoder_train
    # update layer_counter by 2 (2 upsampling layers in autoencoder function)
    layer_counter +=2

'''
DA - XGBoost (noise_factor = 0.5)
Year 1: {'roc_auc': 0.632, 'f1': 0.2987, 'precision': 0.4462, 'recall': 0.225, 'accuracy': 0.6853, 'neg_brier_score': -0.2114}

DA - XGBoost (noise_factor = 0.5, NN depth = 5, n_type = 16, DA CV5 slightly diverge)
Year 1: {'roc_auc': 0.5797, 'f1': 0.209, 'precision': 0.3897, 'recall': 0.143, 'accuracy': 0.6774, 'neg_brier_score': -0.2171}

DA - Logit (noise_factor = 0.02)
Year 1: {'roc_auc': 0.6411, 'f1': 0.0232, 'precision': 0.5125, 'recall': 0.0119, 'accuracy': 0.702, 'neg_brier_score': -0.1997}

DA - Random Forest (noise_factor = 0.02)
Year 1: {'roc_auc': 0.6389, 'f1': 0.1057, 'precision': 0.4685, 'recall': 0.0612, 'accuracy': 0.7011, 'neg_brier_score': -0.1992}

======================= DA with random 20% 0s on nonzero elements =====================
DA0 - XGBoost
Year 1: {'roc_auc': 0.6383, 'f1': 0.3048, 'precision': 0.4465, 'recall': 0.2325, 'accuracy': 0.685, 'neg_brier_score': -0.2102}

DA0 - XGBoost (train epoch=20)
Year 1: {'roc_auc': 0.6363, 'f1': 0.2941, 'precision': 0.4374, 'recall': 0.2222, 'accuracy': 0.6819, 'neg_brier_score': -0.2115}

DA0 - Logit
Year 1: {'roc_auc': 0.6373, 'f1': 0.0388, 'precision': 0.4268, 'recall': 0.0206, 'accuracy': 0.7015, 'neg_brier_score': -0.2002}

DA0, epoch=20, drop NA rows - XGBoost
Year 1: {'roc_auc': 0.633, 'f1': 0.3126, 'precision': 0.433, 'recall': 0.2461, 'accuracy': 0.6691, 'neg_brier_score': -0.219}

DA0, epoch=20, subgroup(supervision_risk_score_first>=6) - XGBoost
Year 1: {'roc_auc': 0.6085, 'f1': 0.3521, 'precision': 0.4633, 'recall': 0.2843, 'accuracy': 0.6416, 'neg_brier_score': -0.2358}

DA0, epoch=20, subgroup(white) - XGBoost
Year 1: {'roc_auc': 0.6112, 'f1': 0.3006, 'precision': 0.4296, 'recall': 0.2323, 'accuracy': 0.6659, 'neg_brier_score': -0.2257}

DA0, epoch=20, subgroup(age_at_release = '23-27', '28-32', '33-37') - XGBoost
Year 1: {'roc_auc': 0.5978, 'f1': 0.3227, 'precision': 0.4302, 'recall': 0.2619, 'accuracy': 0.6433, 'neg_brier_score': -0.2366}

DA0, epoch=20, subgroup(gang_affiliated=1) - XGBoost
Year 1: {'roc_auc': 0.5799, 'f1': 0.4654, 'precision': 0.4991, 'recall': 0.4377, 'accuracy': 0.5664, 'neg_brier_score': -0.2893}

DA0, epoch=20, subgroup(prior_arrest_episodes_ppviolationcharges=1) - XGBoost
Year 1: {'roc_auc': 0.5907, 'f1': 0.3149, 'precision': 0.4365, 'recall': 0.2468, 'accuracy': 0.6404, 'neg_brier_score': -0.2362}

'''
# #
# x = np.array([[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]])
# y = set_random_elements(x, 0.5,999)
#
# z = set_random_elements(train_X, 0.2, 999)
#
# random_idxs0 = np.random.randint(0, x.shape[0], size=2)
# random_idxs1 = np.random.randint(0, x.shape[1], size=2)
# x[random_idxs0, random_idxs1] = 0 # inplace

'''
# Autoencoder - Hyperparemter set up
batch_size = 128
epochs = 50
input_img = Input(shape = (k, 1))

# Compile Autoencoder
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
print(autoencoder.summary())

# Train the model
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
                                    epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

# Predict and compare with ground
pred = autoencoder.predict(train_X)
print(pred - train_ground) # should see small errors

# Training vs Validation Loss Plot
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''




