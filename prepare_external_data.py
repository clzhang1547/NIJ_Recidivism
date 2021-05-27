##################################
# NIJ Recidivism Challenge
# Prepare external data to be merged into NIJ data
# 1. ACS
# 2. GA Crime
# 3. GA THOR
#
# chris zhang 5/27/2021
##################################

import os
import pandas as pd
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 999)
pd.set_option('display.width', 200)
import numpy as np
from aux_functions import *

# Set up a df with a 'group' col
d = pd.Series(range(1, 26), name='group')
d = pd.DataFrame(d)

# ACS data
fp_acs = './from_mason/external features used/ACS'
for x in os.listdir(fp_acs):
    if x in ['population by PUMA.xlsx']:
        pass
    else:
        print('x=%s' % x)
        fp = os.path.join(fp_acs, x)
        df = pd.read_excel(fp)
        d = pd.merge(d, df, on='group', how='left')
# Save clean ACS data
d.to_csv('./data/acs_group.csv', index=False)

# GA Crime data
crime = pd.read_excel('./from_mason/external features used/crime/crime by county linked to puma and group.xlsx')
del crime['Unnamed: 0']
crime.columns = [x.lower() for x in crime.columns]
del crime['puma']
del crime['puma_pop']
# 3 counties fall into multiple PUMA groups:
# Dekalb: 23, 6
# Fulton: 11, 12, 1, 21, 23
# Gwinnett: 12, 14, 6
# Allocate crime evenly to PUMA group (puma pop similar so it should be fine)
dekalb = crime.loc[crime['county']=='Dekalb', ]
dekalb = pd.concat([dekalb]*2)
dekalb['group'] = [6, 23]
dekalb['group_pop'] = [558633, 392950.3333333333]
dekalb['county'] = ['Dekalb_6', 'Dekalb_23']
for c in [x for x in dekalb.columns if x not in ['county', 'group', 'group_pop']]:
    dekalb[c] = dekalb[c]/len(dekalb)

fulton = crime.loc[crime['county']=='Fulton', ]
fulton = pd.concat([fulton]*5)
fulton['group'] = [1, 11, 12, 21, 23]
fulton['group_pop'] = [256480.0, 419755.6666666667, 1154315.333333333, 1154315.333333333, 392950.3333333333]
fulton['county'] = ['Fulton_%s' % x for x in fulton['group']]
for c in [x for x in fulton.columns if x not in ['county', 'group', 'group_pop']]:
    fulton[c] = fulton[c]/len(fulton)

gwinnett = crime.loc[crime['county']=='Gwinnett', ]
gwinnett = pd.concat([gwinnett]*3)
gwinnett['group'] = [6, 12, 14]
gwinnett['group_pop'] = [558633.0, 1154315.333333333, 537332.6666666666]
gwinnett['county'] = ['Gwinnett_%s' % x for x in gwinnett['group']]
for c in [x for x in fulton.columns if x not in ['county', 'group', 'group_pop']]:
    gwinnett[c] = gwinnett[c]/len(gwinnett)
# get cleaned county crime data
crime = crime[~crime['county'].isin(['Dekalb', 'Fulton', 'Gwinnett'])]
crime = crime.append(dekalb)
crime = crime.append(fulton)
crime = crime.append(gwinnett)
# get PUMA group crime
cols_crime = ['murder', 'rape', 'robbery','assault', 'burglary', 'larceny', 'vehicle theft']
crime_group = crime[cols_crime + ['group']].groupby('group').sum().reset_index()
pop_group = pd.read_excel('./from_mason/external features used/ACS/population by group.xlsx')
crime_group = pd.merge(crime_group, pop_group, on='group', how='left')
for c in cols_crime:
    crime_group['%s_per_k' % c] = crime_group[c]/crime_group['PWGTP']*1000
del crime_group['PWGTP']
# Save clean crime data
crime_group.to_csv('./data/crime_group.csv', index=False)

# THOR data
thor = pd.read_csv('./from_mason/external features used/THOR/georgia_thor.csv')
thor.columns = [x.lower().strip() for x in thor.columns]
for c in thor.columns:
    thor[c] = [x.strip() for x in thor[c]]
thor['n_fac'] = 1 # number of facilities
# THOR Dekalb
thor_dekalb = thor.loc[thor['county']=='Dekalb', ]
thor_dekalb = pd.concat([thor_dekalb]*2)
thor_dekalb['group'] = [6, 6, 6, 6, 23, 23, 23, 23,]
thor_dekalb['county'] = ['Dekalb_%s' % x for x in thor_dekalb['group']]
thor_dekalb['n_fac'] = 1/2
# THOR Fulton
thor_fulton = thor.loc[thor['county']=='Fulton', ]
n_fac_fulton = len(thor_fulton)
thor_fulton = pd.concat([thor_fulton]*5) # 11, 12, 1, 21, 23
thor_fulton['group'] = [11]*n_fac_fulton + [12]*n_fac_fulton + [1]*n_fac_fulton + [21]*n_fac_fulton + [23]*n_fac_fulton
thor_fulton['county'] = ['Fulton_%s' % x for x in thor_fulton['group']]
thor_fulton['n_fac'] = 1/5
# THOR Gwinnett
thor_gwinnett = thor.loc[thor['county']=='Gwinnett', ]
n_fac_gwinett = len(thor_gwinnett)
thor_gwinnett = pd.concat([thor_gwinnett]*3) # 12, 14, 6
thor_gwinnett['group'] = [12]*n_fac_gwinett + [14]*n_fac_gwinett + [6]*n_fac_gwinett
thor_gwinnett['county'] = ['Gwinnett%s' % x for x in thor_gwinnett['group']]
thor_gwinnett['n_fac'] = 1/3
# get clean county THOR data
thor = thor[~thor['county'].isin(['Dekalb', 'Fulton', 'Gwinnett'])]
for df in [thor_dekalb, thor_fulton, thor_gwinnett]:
    df = df[thor.columns]
    thor = thor.append(df)
# get a county-PUMA group mapping from crime data
thor = pd.merge(thor, crime[['county', 'group']], on='county',  how='left')
thor_group_male = thor[thor['gender']=='Male'][['n_fac', 'group']].groupby('group').sum()
thor_group_male = thor_group_male.rename(columns={'n_fac':'n_fac_male'})
thor_group_female = thor[thor['gender']=='Female'][['n_fac', 'group']].groupby('group').sum()
thor_group_female = thor_group_female.rename(columns={'n_fac':'n_fac_female'})
thor_group_both= thor[thor['gender']=='Male/Female'][['n_fac', 'group']].groupby('group').sum()
thor_group_both = thor_group_both.rename(columns={'n_fac':'n_fac_both'})
thor_group = thor_group_male.join(thor_group_female)
thor_group = thor_group.join(thor_group_both)
thor_group = thor_group.fillna(0)
thor_group = thor_group.reset_index()

# Save clean THOR data
thor_group.to_csv('./data/thor_group.csv', index=False)