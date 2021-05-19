##################################
# NIJ Recidivism Challenge
# Denoise Autoencoder - Implementation with NIJ data
# chris zhang 5/18/2021
#
# TODO [done]: check DA loss graph on denoising x_train_noisy <- noise factor=0.01 works for NIJ data
# ^noise factor=0.02 divegence occurs, 0.1, 0.5 bad
# TODO [done]: add constant xvars so #cols = 4X, 8X allowing encoding/decoding flexibility to lower dim
# TODO: set noise a la Beaulieu-Jones 2016: randomly set to 0 for 20% values of input matrix
# TODO: Year 1 results with 128>>64 xvars and 0.02 noise get similar/worse perf vs ML. Try 32 xvars, random 0 as noise
##################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

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

# Read in data
# d will be purely train data (no id, no yvars)
d = read_and_clean_raw(fp='data/nij/NIJ_s_Recidivism_Challenge_Training_Dataset.csv')
cols_ys = ['recidivism_within_3years', 'recidivism_arrest_year1', 'recidivism_arrest_year2', 'recidivism_arrest_year3']
yvar = 'recidivism_arrest_year1'
train_labels = d[yvar].astype(int)
d = d[[x for x in d.columns if x not in cols_ys]]

# Fill in NAs
print(d.isna().sum())
na_count = d.isna().sum()
na_cols = na_count[na_count>0].index
for c in na_cols:
    d[c] = fill_na(d[c])
# check dtype, set puma to str
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
# supervision activity cols, to be excl. from year 1 model
cols_sup_act = ['drugtests_other_positive', 'drugtests_meth_positive', 'avg_days_per_drugtest',
                'violations_failtoreport', 'jobs_per_year', 'program_unexcusedabsences', 'residence_changes',
                'program_attendances', 'drugtests_thc_positive', 'delinquency_reports',
                'drugtests_cocaine_positive', 'violations_instruction', 'violations_movewithoutpermission',
                'percent_days_employed', 'employment_exempt', 'violations_electronicmonitoring']
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
for i in range(6):
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

# Autoencoder - Hyperparemter set up
batch_size = 128
epochs = 50
input_img = Input(shape = (k, 1))

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

# def autoencoder(input_img):
#     # Note - k should be dividable by pool_size multiple times, as set by conv structure
#     #encoder
#     #input = k x 1 (wide and thin)
#     conv1 = Conv1D(32, (3), activation='relu', padding='same')(input_img) # k x 1 x 32
#     pool1 = MaxPooling1D(pool_size=2)(conv1) # k/2 x 1 x 32
#     conv2 = Conv1D(64, (3), activation='relu', padding='same')(pool1) # k/2 x 1 x 64
#     pool2 = MaxPooling1D(pool_size=2)(conv2) # k/4 x 1 x 64
#     conv3 = Conv1D(128, (3), activation='relu', padding='same')(pool2) # k/4 x 1 x 128
#     pool3 = MaxPooling1D(pool_size=2)(conv3) # k/8 x 1 x 128
#     conv4 = Conv1D(256, (3), activation='relu', padding='same')(pool3) # k/8 x 1 x 256
#     pool4 = MaxPooling1D(pool_size=2)(conv4) # k/16 x 1 x 256
#     conv5 = Conv1D(512, (3), activation='relu', padding='same')(pool4) # k/16 x 1 x 512
#
#     #decoder
#     conv6 = Conv1D(512, (3), activation='relu', padding='same')(conv5) # k/16 x 1 x 512
#     up1 = UpSampling1D(2)(conv6) # k/8 x 1 x 512
#     conv7 = Conv1D(256, (3), activation='relu', padding='same')(up1) # k/8 x 1 x 256
#     up2 = UpSampling1D(2)(conv7) # k/4 x 1 x 64
#     decoded = Conv1D(1, (3,), activation='sigmoid', padding='same')(up2) # k/4 x 1 x 1
#     return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
print(autoencoder.summary())

# Train the model
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
                                    epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

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

# predict and compare with ground
pred = autoencoder.predict(train_X)
print(pred - train_ground) # should see small errors

# Add noise to train/valid data
noise_factor = 0.02 # 0.5 on NIJ data would cause valid loss diverging up, 0.01 is good for convergence
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)

# Denoise Autoencoder Hyperparameters
batch_size = 128
epochs = 50
input_img = Input(shape = (k, 1))

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = 'rmsprop')
# Train model - DA (note x_train_noisy and train_X are same shape)
autoencoder_train = autoencoder.fit(x_train_noisy, train_X, batch_size=batch_size,epochs=epochs,
                                    verbose=1,validation_data=(x_valid_noisy, valid_X))

# Get intermediate layer (phenotype) from trained model
layer_name = 'up_sampling1d_8'
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)
d_noisy = d + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=d.shape)
intermediate_output = intermediate_layer_model.predict(d_noisy) # N x k/4 x 128, get phenotype for entire dataset d
phenotype = Conv1D(1, (3,), activation='sigmoid', padding='same')(intermediate_output)  # N x k/4 x 1

# XGBoost for phenotype and train_labels
clf = xgboost.XGBClassifier(objective='binary:logistic', use_label_encoder=False) # one hot encoding
clf = LogisticRegression(max_iter=1000)
clf = RandomForestClassifier()
scores = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'neg_brier_score']
# Model - Year 1
y = train_labels
X = phenotype.numpy()[0]
dct_score = {}
for s in scores:
    dct_score[s] = round(cross_val_score(clf, X, y, cv=5, scoring=s).mean(), 4)
    print('CV score completed -- %s' % s)
print(dct_score)

'''
DA - XGBoost
Year 1: {'roc_auc': 0.6054, 'f1': 0.2738, 'precision': 0.4233, 'recall': 0.2036, 'accuracy': 0.679, 'neg_brier_score': -0.2174}

DA - Logit
Year 1: {'roc_auc': 0.6362, 'f1': 0.0059, 'precision': 0.6143, 'recall': 0.003, 'accuracy': 0.7022, 'neg_brier_score': -0.2013}
'''




