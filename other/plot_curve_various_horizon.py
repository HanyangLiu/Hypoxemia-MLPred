import matplotlib.pyplot as plt
import seaborn as sns
from file_config.config import config
from utils.utility_analysis import line_search_best_metric
from ast import literal_eval
import numpy as np
import pandas as pd

# load saved pickle
result_table = pd.read_pickle('../data/result/various_pred_win.pkl')

# win = 10
# y_prob = result_table.loc[win, 'y_prob']
# y_test = result_table.loc[win, 'y_test']
# _, y_pred = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)
#
# win_count = np.zeros(win)
# win_corr = np.zeros(win)
# win_first = np.zeros(win)
#
# if_correct = y_pred == y_test
# for ind, _ in enumerate(y_test):
#     if ind < win:
#         continue
#     if y_test[ind] == 1 and y_test[ind + 1] == 0:
#         win_arr = if_correct[ind - win + 1: ind + 1]
#         win_count += win_arr
#         for i, a in enumerate(win_arr):
#             if np.sum(win_arr) - np.sum(win_arr[0:i]) == win - i:
#                 win_corr[i] += 1
#         for i, a in enumerate(win_arr):
#             if np.sum(win_arr[0:i]) == 0 and win_arr[i] == 1:
#                 win_first[i] += 1
#
#

lw = 1

# plot PRC
plt.figure(figsize=[6, 5])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    plt.plot(result_table.loc[win_size]['rec'],
             result_table.loc[win_size]['prec'],
             color=color_ls[ind],
             label="{} min window, AUC={:.3f}".format(win_size, result_table.loc[win_size, 'prc']),
             linewidth=lw
             )

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Sensitivity", fontweight='bold')
plt.ylabel("Precision", fontweight='bold')

# plt.title('Precision-Recall Curve (PRC) Analysis')
plt.legend(loc='upper right')
plt.savefig('../data/result/realtime_prc.pdf')
plt.show()

# plot ROC
plt.figure(figsize=[6, 5])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    plt.plot(result_table.loc[win_size]['fpr'],
             result_table.loc[win_size]['tpr'],
             color=color_ls[ind],
             label="{} min window, AUC={:.3f}".format(win_size, result_table.loc[win_size, 'roc']),
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
plt.figure(figsize=[6, 5])
color_ls = sns.color_palette("coolwarm_r", len(result_table))

for ind, win_size in enumerate(result_table.index):
    rec = result_table.loc[win_size]['rec']
    prec = result_table.loc[win_size]['prec']
    alarm_rate = result_table.loc[win_size]['pos_rate'] * rec / prec * 60
    plt.plot(alarm_rate,
             rec,
             color=color_ls[ind],
             label="{} min window, sensitivity={:.3f}".format(win_size, rec[np.isclose(alarm_rate, 1, atol=0.001)][0]),
             linewidth=lw
             )

plt.plot(1.0 * np.ones(10),
         np.linspace(-0.05, 1.05, 10),
         linestyle=(0, (9, 10)),
         color='black',
         label='Alarm rate=1',
         linewidth=0.8
         )

plt.xlim([-0.05, 3.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("Alarm rate (number of alarms per hour per patient)", fontweight='bold')
plt.ylabel("Sensitivity", fontweight='bold')

# plt.title('Sensitivity vs Alarm Rate')
plt.legend(loc='lower right')
plt.savefig('../data/result/realtime_sens_alarm.pdf')
plt.show()

