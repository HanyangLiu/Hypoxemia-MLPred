import matplotlib.pyplot as plt
import seaborn as sns
from file_config.config import config
from utils.utility_analysis import line_search_best_metric
from ast import literal_eval
import numpy as np
import pandas as pd

# load saved pickle
result_table = pd.read_pickle('/storage1/lu/Active/Hanyang/hybrid-inference/data_hypoxemia/result/lstm_horizon.pkl')
result_table = result_table.sort_values(by='horizon').set_index('horizon')

lw = 1.5
num_interp = 200
# plt.style.use('ggplot')

# plot PRC
plt.figure(figsize=[5, 4])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    mean_prec = np.linspace(0, 1, num_interp)
    interp_rec = np.interp(mean_prec, result_table.loc[win_size, 'prec'], result_table.loc[win_size, 'rec'])
    interp_rec[0] = 1.0
    plt.plot(interp_rec,
             mean_prec,
             color=color_ls[ind],
             label="horizon={} min, AUC={:.4f}".format(win_size, result_table.loc[win_size, 'prc']),
             linewidth=lw
             )

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Sensitivity", fontweight='bold')
plt.ylabel("Precision", fontweight='bold')

# plt.title('Precision-Recall Curve (PRC) Analysis')
plt.legend(loc='lower left')
plt.savefig('../data/result/realtime_prc.pdf')
plt.show()

# plot ROC
plt.figure(figsize=[5, 4])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    mean_fpr = np.linspace(0, 1, num_interp)
    interp_tpr = np.interp(mean_fpr, result_table.loc[win_size]['fpr'], result_table.loc[win_size]['tpr'])
    interp_tpr[0] = 0.0
    plt.plot(mean_fpr,
             interp_tpr,
             color=color_ls[ind],
             label="horizon={} min, AUC={:.4f}".format(win_size, result_table.loc[win_size, 'roc']),
             linewidth=lw
             )

plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=0.5)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Flase Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')

# plt.title('Receiver Operating Characteristic (ROC) Analysis')
plt.legend(loc='lower right')
plt.savefig('../data/result/realtime_roc.pdf')
plt.show()


# plot sensitivity-alarm
plt.figure(figsize=[5, 4])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    rec = result_table.loc[win_size]['rec']
    prec = result_table.loc[win_size]['prec']
    mean_prec = np.linspace(0, 1, 500)
    interp_rec = np.interp(mean_prec, prec, rec)
    mean_prec[0] = 1e-5
    interp_rec[0] = 1.0
    alarm_rate = result_table.loc[win_size]['pos_rate'] * interp_rec / mean_prec * 60
    ar = alarm_rate[np.isclose(interp_rec, 0.8, atol=0.05)][0]
    plt.plot(
             interp_rec,
             alarm_rate,
             color=color_ls[ind],
             label="horizon={} min, alert rate={:.2f}".format(win_size, ar),
             linewidth=lw
             )

plt.plot(0.8 * np.ones(10),
         np.linspace(-0.05, 10.05, 10),
         linestyle=(0, (9, 10)),
         color='black',
         label='Sensitivity cut=0.8',
         linewidth=0.8
         )

plt.ylim([-0.02, 0.42])
plt.xlim([-0.05, 1.05])
plt.ylabel("Alert Rate (# Alert/hr)", fontweight='bold')
plt.xlabel("Sensitivity", fontweight='bold')

# plt.title('Sensitivity vs Alarm Rate')
plt.legend(loc='upper left')
plt.savefig('../data/result/realtime_alarm_sens.pdf')
plt.show()

