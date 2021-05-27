import matplotlib.pyplot as plt
import seaborn as sns
from file_config.config import config
from utils.utility_analysis import line_search_best_metric
from ast import literal_eval
import numpy as np
import pandas as pd
import scikitplot as skplt


def get_df_mean(df, column):
    values = np.array(df[column].to_list())
    mean_values = np.mean(values, axis=0)
    return mean_values


def plot_model_curves(result_table, name_converter):
    mean_table = pd.DataFrame(columns=['model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])

    models = result_table['model'].unique()
    for model in models:
        df = result_table[result_table['model'] == model]
        mean_fpr = np.linspace(0, 1, 1000)
        mean_prec = np.linspace(0, 1, 1000)
        tprs = []
        recs = []

        for ind in df.index:
            fpr = df.loc[ind, 'fpr']
            tpr = df.loc[ind, 'tpr']
            rec = df.loc[ind, 'rec']
            prec = df.loc[ind, 'prec']
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            interp_rec = np.interp(mean_prec, prec, rec)
            interp_rec[0] = 1.0
            recs.append(interp_rec)

        mean_tpr = np.mean(tprs, axis=0)
        mean_rec = np.mean(recs, axis=0)

        mean_table = mean_table.append({
            'model': model,
            'fpr': mean_fpr,
            'tpr': mean_tpr,
            'roc': get_df_mean(df, 'roc'),
            'prec': mean_prec,
            'rec': mean_rec,
            'prc': get_df_mean(df, 'prc'),
            'y_test': get_df_mean(df, 'y_test'),
            'y_prob': get_df_mean(df, 'y_prob'),
            'pos_rate': get_df_mean(df, 'pos_rate')
        }, ignore_index=True)

    # plot PRC
    plt.figure(figsize=[6, 5])
    # color_ls = sns.color_palette("coolwarm_r", len(result_table))

    for ind in mean_table.index:
        label = "{}, AUC={:.3f}".format(name_converter[mean_table.loc[ind, 'model']], mean_table.loc[ind, 'prc'])
        plt.plot(mean_table.loc[ind]['rec'],
                 mean_table.loc[ind]['prec'],
                 # color=color_ls[ind],
                 label=label,
                 # linewidth=1
                 )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Sensitivity")
    plt.ylabel("Precision")

    plt.title('Precision-Recall Curve (PRC) Analysis')
    plt.legend(loc='upper right')
    plt.show()

    # plot ROC
    plt.figure(figsize=[6, 5])
    # color_ls = sns.color_palette("coolwarm_r", len(result_table))

    for ind in mean_table.index:
        label = "{}, AUC={:.3f}".format(name_converter[mean_table.loc[ind, 'model']], mean_table.loc[ind, 'roc'])
        plt.plot(mean_table.loc[ind]['fpr'],
                 mean_table.loc[ind]['tpr'],
                 # color=color_ls[ind],
                 label=label,
                 # linewidth=1
                 )

    plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=0.5)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Flase Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title('Receiver Operating Characteristic (ROC) Analysis')
    plt.legend(loc='lower right')
    plt.show()

    # plot sensitivity-alarm
    plt.figure(figsize=[6, 5])
    # color_ls = sns.color_palette("coolwarm_r", len(result_table))

    for ind in mean_table.index:
        rec = mean_table.loc[ind]['prec']
        prec = mean_table.loc[ind]['rec']
        alarm_rate = mean_table.loc[ind]['pos_rate'] * rec / prec * 60
        plt.plot(alarm_rate,
                 rec,
                 # color=color_ls[ind],
                 # label="{}, sensitivity={:.3f}".format(name_converter[mean_table.loc[ind, 'model']], rec[np.isclose(alarm_rate, 1, atol=0.1)][0]),
                 label="{}".format(name_converter[mean_table.loc[ind, 'model']]),
                 # linewidth=1
                 )

    plt.plot(1.0 * np.ones(10),
             np.linspace(-0.05, 1.05, 10),
             linestyle=(0, (9, 10)),
             color='black',
             label='Alarm rate=1',
             linewidth=0.8
             )

    plt.xlim([-0.05, 2.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Alarm rate (number of alarms per hour per patient)", fontweight='bold')
    plt.ylabel("Sensitivity", fontweight='bold')

    plt.title('Sensitivity vs Alarm Rate')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    result_table = pd.read_pickle('../data/result/realtime_gbtree.pkl')
    cnnlstm = pd.read_pickle('../data/result/realtime_cnnlstm.pkl')
    cnnlstm.loc[0, 'model'] = 'CNN-LSTM'
    cnnlstm = cnnlstm.append(result_table, ignore_index=True)
    cnnlstm.loc[1, 'model'] = "CatBoostClassifier"
    table = pd.read_pickle('../data/result/realtime_lstm.pkl')
    result_table = cnnlstm.append(table, ignore_index=True)



    name_converter = {
        "CatBoostClassifier": "GBM",
        "LogisticRegression": "LR",
        "RandomForestClassifier": "RF",
        "CNN-LSTM": "CNN-LSTM",
        "LSTM": "LSTM"
    }

    plot_model_curves(result_table, name_converter)


