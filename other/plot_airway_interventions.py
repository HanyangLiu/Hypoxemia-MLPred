
from utils.utility_analysis import *
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_raw = pd.read_csv('../data/raw_data/static_updated.csv')
df_static = pd.read_csv('../data/data_frame/static_dataframe.csv')
df_dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')

airway_list = df_raw['Airway_Event1'].unique()
df = df_raw[df_raw['Airway_Event1'] == airway_list[0]][0:10]
for airway in airway_list:
    if airway == airway_list[0]:
        continue
    df = df.append(df_raw[df_raw['Airway_Event1'] == airway][0:10])
index = df.index
df = df.reindex(columns=['Airway_Event1'] + df.columns.to_list())
df.to_csv('../data/results/airway_samples.csv')

# label assignment (according to imputed SpO2)
print('Assigning labels...')
imputer = DataImputation()
df_static = imputer.impute_static_dataframe(df_static)
df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
label_assign = LabelAssignment(hypoxemia_thresh=90,
                               hypoxemia_window=10,
                               prediction_window=5)
static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
positive_pids = label_assign.get_positive_pids(static_label)
print('Done.')

# get subgroup pids
subgroup_pids = PatientFilter(df_static=df_static,
                              mode='exclude',
                              include_icd=None,
                              exclude_icd9=['745', '746', '747'],
                              exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()

print('Positive Patient:', len(set(subgroup_pids) & set(positive_pids)), '/', len(subgroup_pids))
print('Before trimming:', len(positive_pids), '/', len(df_static))
print('Trimmed cases:', len(df_static) - len(subgroup_pids))
pos_rate = len(set(subgroup_pids) & set(positive_pids)) / len(subgroup_pids)
print('Positive rate:', pos_rate)

# select features with pid in subgroup as data matrix, and split into training and test set
selected_idx = subgroup_pids
X = df_raw.iloc[selected_idx, 1:]
y = static_label.loc[selected_idx, 'label']
X_pos = X[y == 1]

distr_all = X['Airway_Event1'].value_counts()
distr_all = distr_all/len(X)
distr_all['No airway intervention'] = 1 - distr_all.sum()

distr_pos = X_pos['Airway_Event1'].value_counts()
distr_pos = distr_pos/len(X_pos)
distr_pos = distr_pos.reindex(index=distr_all.index)
distr_pos['No airway intervention'] = 1 - distr_pos.sum()

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot()

# set width of bar
barWidth = 0.3

# set height of bar
bars1 = list(distr_all.values)
bars2 = list(distr_pos.values)

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='All patients (Cardiac cases excluded)')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Hypoxemia patients')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(bars1))], list(distr_pos.index))
fig.autofmt_xdate(rotation=45)
plt.xlabel('Type of Interventions', fontweight='bold')
plt.ylabel('Probability Distribution', fontweight='bold')

# Create legend & Show graphic
plt.legend(loc='upper right')
plt.show()

