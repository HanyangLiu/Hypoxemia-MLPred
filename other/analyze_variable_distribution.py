import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric, au_prc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
from train_catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import pickle
import sys
import shap

# mpl.rcParams['agg.path.chunksize'] = 10000

df_static = pd.read_csv('../data/data_frame/static_dataframe.csv')
df_dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')

# label assignment (according to imputed SpO2)
print('Assigning labels...')
imputer = DataImputation()
df_static = imputer.impute_static_dataframe(df_static)
df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
path_sta_label = '../data/label/static_label.pkl'
path_dyn_label = '../data/label/dynamic_label.pkl'
label_assign = LabelAssignment(hypoxemia_thresh=90,
                               hypoxemia_window=10,
                               prediction_window=5)

# print('Assigning labels...')
# static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
# static_label.to_pickle(path_sta_label)
# dynamic_label.to_pickle(path_dyn_label)

static_label = pd.read_pickle(path_sta_label)
dynamic_label = pd.read_pickle(path_dyn_label)
positive_pids = label_assign.get_positive_pids(static_label)
print('Done.')
# get subgroup pids
subgroup_pids = PatientFilter(df_static=df_static,
                              mode='exclude',
                              include_icd=['J96.', 'J98.', '519.', '518.', '277.0', 'E84', 'Q31.5', '770.7',
                                           'P27.1', '490', '491', '492', '493', '494', '495', '496', 'P27.8',
                                           'P27.9', 'J44', 'V46.1', 'Z99.1'],  # High-risk group
                              exclude_icd9=['745', '746', '747'],
                              exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()

plt.figure()
plt.hist(df_static[df_static['pid'].isin(subgroup_pids)]['Airway_1'].values, normed=True, bins=10, label='All patients', alpha=0.7)
aw1 = df_static[df_static['pid'].isin(set(subgroup_pids) & set(positive_pids))]['Airway_1'].values
plt.hist(aw1, normed=True, bins=9, label='Hypoxemic patients', alpha=0.7)
plt.xlabel('labels of airway interventions')
plt.ylabel('normalized frequency')
plt.legend()
plt.title('Distribution of Airway Intervention Types')
plt.show()
at1 = df_static[df_static['pid'].isin(set(subgroup_pids) & set(positive_pids))]['Airway_1_Time'].values
aw2 = df_static[df_static['pid'].isin(set(subgroup_pids) & set(positive_pids))]['Airway_2'].values

df_dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')
df = df_dynamic[df_dynamic['pid'] == 60]


fig = plt.figure(figsize=(10, 12))
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

num = 11
hight = 1/(num + 1)

# SpO2
ax = fig.add_axes([hight, 0.05 + hight * (num - 1), 0.85, hight],
                  xticks=[],
                  ylabel='$SpO_2$\n(%)'
                  )
ax.plot(df['ts'].values, df['SpO2'].values, marker='.', markersize=5, linewidth=0.5)
# ax.plot(df['ts'].values, df[dynamic_label['if']].values)
ax.plot(df['ts'].values, 90 * np.ones(len(df)), '--', linewidth=1)

# FiO2
ax = fig.add_axes([hight, 0.05 + hight * (num - 2), 0.85, hight],
                  xticks=[],
                  ylabel='FiO2\n(%)')
ax.plot(df['ts'].values, df['FiO2'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5)

# ETCO2
ax = fig.add_axes([hight, 0.05 + hight * (num - 3), 0.85, hight],
                  xticks=[],
                  ylabel='$ETCO_2$\n(mm Hg)',
                  ylim=[-5, 35]
                  )
ax.plot(df['ts'].values, df['ETCO2'].values, marker='.', markersize=5, linewidth=0.5)

# Invasive BP
ax = fig.add_axes([hight, 0.05 + hight * (num - 4), 0.85, hight],
                  xticks=[],
                  ylabel='Invasive BP\n(mm Hg)'
                  )
ax.plot(df['ts'].values, df['invSystolic'].values, color='tab:blue', marker='.', markersize=5, label='Invasive Systolic', linewidth=0.5)
ax.plot(df['ts'].values, df['invMeanBP'].values, color='tab:blue', marker='^', markersize=5, label='Invasive Mean BP', linewidth=0.5)
ax.plot(df['ts'].values, df['invDiastolic'].values, color='tab:blue', marker='*', markersize=5, label='Invasive Diastolic', linewidth=0.5)
ax.legend()

# Non-inv BP
ax = fig.add_axes([hight, 0.05 + hight * (num - 5), 0.85, hight],
                  xticks=[],
                  ylabel='BP\n (mm Hg)',
                  ylim=[20, 170])
ax.plot(df['ts'].values, df['Systolic'].values, color='tab:blue', marker='.', markersize=5, label='Systolic', linewidth=0.5)
ax.plot(df['ts'].values, df['MeanBP'].values, color='tab:blue', marker='^', markersize=5, label='Mean BP', linewidth=0.5)
ax.plot(df['ts'].values, df['Diastolic'].values, color='tab:blue', marker='*', markersize=5, label='Diastolic', linewidth=0.5)
ax.legend()

# RespRate
ax = fig.add_axes([hight, 0.05 + hight * (num - 6), 0.85, hight],
                  xticks=[],
                  ylabel='Resp. Rate\n(per min)')
ax.plot(df['ts'].values, df['RespRate'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5)

# PIP/PEEP
ax = fig.add_axes([hight, 0.05 + hight * (num - 7), 0.85, hight],
                  xticks=[],
                  ylabel='PIP/PEEP\n(cm $H_2O$)')
ax.plot(df['ts'].values, df['PIP'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5, label='PIP')
ax.plot(df['ts'].values, df['PEEP'].values, color='tab:blue', marker='*', markersize=8, linewidth=0.5, label='PEEP')
ax.legend()

# Air Flow
ax = fig.add_axes([hight, 0.05 + hight * (num - 8), 0.85, hight],
                  xticks=[],
                  ylabel='Air Flow\n(l/min)',
                  ylim=[1000, 5000]
                  )
ax.plot(df['ts'].values, df['AirFlow'].values, color='tab:blue', marker='.', markersize=5, label='Air Flow', linewidth=0.5)
ax.plot(df['ts'].values, df['O2Flow'].values, color='tab:blue', marker='^', markersize=5, label='$O_2 Flow$', linewidth=0.5)
ax.plot(df['ts'].values, df['N2OFlow'].values, color='tab:blue', marker='*', markersize=5, label='$N_2O$ Flow', linewidth=0.5)
ax.legend()

# Tidal Volume
ax = fig.add_axes([hight, 0.05 + hight * (num - 9), 0.85, hight],
                  xticks=[],
                  ylabel='Tidal Volume\n(l)')
ax.plot(df['ts'].values, df['TidalVolume'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5)

# Temp
ax = fig.add_axes([hight, 0.05 + hight * (num - 10), 0.85, hight],
                  xticks=[],
                  ylabel='Temperature\n($^o F$)')
ax.plot(df['ts'].values, df['Temp'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5)

# HR/pulse
ax = fig.add_axes([hight, 0.05 + hight * (num - 11), 0.85, hight],
                  ylabel='HR/Pulse\n(per min)')
ax.plot(df['ts'].values, df['HR'].values, color='tab:blue', marker='.', markersize=5, linewidth=0.5, label='Heart Rate')
ax.plot(df['ts'].values, df['Pulse'].values, color='tab:purple', marker='*', markersize=8, linewidth=0.5, label='Pulse')
ax.legend()

fig.show()

plt.figure(figsize=(6, 2))
plt.plot(df['ts'].values, df['HR'].values - 70, linewidth=2, label='Heart Rate')
plt.plot(df['ts'].values, df['Pulse'].values - 60, linewidth=2, label='Pulse')
plt.plot(df['ts'].values, 0.3*df['TidalVolume'].values, linewidth=2)
plt.plot(df['ts'].values, df['PIP'].values, linewidth=2, label='PIP')
plt.plot(df['ts'].values, (df['SpO2'].values - 80) * 3 - 40, linewidth=2)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig('../data/result/vitals.pdf')
plt.show()

