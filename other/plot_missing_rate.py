from utils.utility_analysis import *
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
import numpy as np
import matplotlib.pyplot as plt


df_dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')
filling_ratio = feature_value_filling_ratio(df_dynamic)
feat_name = ['invDiastolic', 'invMeanBP', 'invSystolic', 'HR', 'Diastolic', 'MeanBP', 'Systolic', 'SpO2',
             'RespRate', 'PEEP', 'PIP', 'FiO2', 'TidalVolume', 'Pulse', 'ETCO2', 'O2Flow', 'AirFlow',
             'N2OFlow', 'Temp', 'coreTemp']
# interpolate if gap is less than 10 timestep
mask = df_dynamic[feat_name].copy()
df_dynamic_imp = df_dynamic.copy()
for column in feat_name:
    df = pd.DataFrame(df_dynamic_imp[column])
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask[column] = (df.groupby('new')['ones'].transform('count')
                    < 20) | df_dynamic_imp[column].notnull()
df_dynamic_imp[feat_name] = df_dynamic_imp[feat_name].interpolate().bfill()[mask]
df_dynamic_imp[['FiO2', 'N2OFlow']] = df_dynamic_imp[['pid', 'FiO2', 'N2OFlow']].groupby(['pid']).ffill()
filling_ratio_imp = feature_value_filling_ratio(df_dynamic_imp)

fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot()

# set width of bar
barWidth = 0.3

# set height of bar
bars1 = list(filling_ratio.values())[3:23]
bars2 = list(filling_ratio_imp.values())[3:23]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='Before')
plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='After')

# Add xticks on the middle of the group bars
plt.xticks([r + barWidth for r in range(len(bars1))], list(filling_ratio.keys())[3:])
fig.autofmt_xdate(rotation=45)
plt.xlabel('Input Variable', fontweight='bold')
plt.ylabel('1 - Missing Rate', fontweight='bold')

# Create legend & Show graphic
plt.legend()
plt.show()



