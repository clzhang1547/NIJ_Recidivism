import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score

pred = pd.read_excel('./_management/_submission/Year1/AMERICAN INSTITUTES FOR RESEARCH_1YearForecast.xlsx')
# Read in "test data"
# diff between test1 and test2 gives recd of Year1,
# diff between test2 and test3 gives recd of Year2,
test2 = pd.read_csv('./data/nij/NIJ_s_Recidivism_Challenge_Test_Dataset2.csv')
test3 = pd.read_csv('./data/nij/NIJ_s_Recidivism_Challenge_Test_Dataset3.csv')
recd = list(set(test2.ID) - set(test3.ID))
recd = pd.Series(recd, name='ID')
recd = pd.DataFrame(recd)
recd['true'] = 1
pred = pd.merge(pred, recd, on='ID', how='left')
pred.loc[pred['true'].isna(), 'true'] = 0

pred['pred'] = np.where(pred['Probability']>=0.5, 1, 0)

score = {}
score['brier'] = brier_score_loss(pred['true'], pred['Probability'])
score['accuracy'] = accuracy_score(pred['true'], pred['pred'])
score['precision'] = precision_score(pred['true'], pred['pred'])
score['recall'] = recall_score(pred['true'], pred['pred'])
score['f1'] = f1_score(pred['true'], pred['pred'])
score['roc'] = roc_auc_score(pred['true'], pred['Probability'])
print(score)
print('-'*150)


###############
# Plot
###############
import matplotlib.pyplot as plt
import seaborn as sns

# phat by true label
sns.kdeplot(data=pred[pred['true']==0]['Probability'], label='True Year 1 = 0')
sns.kdeplot(data=pred[pred['true']==1]['Probability'], label='True Year 1 = 1')
plt.legend()
plt.show()

# phat by true/false predictions
sns.kdeplot(data=pred[(pred['true']==1) & (pred['pred']==1)]['Probability'], label='True Positives')
sns.kdeplot(data=pred[(pred['true']==0) & (pred['pred']==0)]['Probability'], label='True Negatives')
sns.kdeplot(data=pred[(pred['true']==0) & (pred['pred']==1)]['Probability'], label='False Positives', linestyle='--')
sns.kdeplot(data=pred[(pred['true']==1) & (pred['pred']==0)]['Probability'], label='False Negatives', linestyle='--')
plt.legend()
plt.show()


