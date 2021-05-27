import matplotlib.pyplot as plt
import seaborn as sns
from file_config.config import config
from utils.utility_analysis import line_search_best_metric
from ast import literal_eval
import numpy as np
import pandas as pd
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_df_mean(df, column):
    values = np.array(df[column].to_list())
    mean_values = np.mean(values, axis=0)
    return mean_values


def plot_model_curves(result_table, name_converter):
    mean_table = pd.DataFrame(columns=['model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])

    models = result_table['model'].unique()
    for model in models:
        df = result_table[result_table['model'] == model]
        mean_fpr = np.linspace(0, 1, 500)
        mean_prec = np.linspace(0, 1, 500)
        tprs = []
        recs = []
        rocs = []
        prcs = []

        for ind in df.index:
            fpr = df.loc[ind, 'fpr']
            tpr = df.loc[ind, 'tpr']
            roc = df.loc[ind, 'roc']
            rec = df.loc[ind, 'rec']
            prec = df.loc[ind, 'prec']
            prc = df.loc[ind, 'prc']
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            rocs.append(roc)
            interp_rec = np.interp(mean_prec, prec, rec)
            # interp_rec[0] = 1.0
            recs.append(interp_rec)
            prcs.append(prc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_rec = np.mean(recs, axis=0)
        mean_roc, low_roc, high_roc = mean_confidence_interval(rocs)
        mean_prc, low_prc, high_prc = mean_confidence_interval(prcs)

        print('----------------------------')
        print('ROC of', model)
        print('ROC:{:.4f} ({:.4f}, {:.4f})'.format(mean_roc, low_roc, high_roc))
        print('ROC:{:.4f} ({:.4f}, {:.4f})'.format(mean_prc, low_prc, high_prc))

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

    # mean_table = mean_table.sort_values(by=['prc'], ignore_index=True, ascending=False)

    # plot PRC
    # plt.style.use('ggplot')
    plt.figure(figsize=[5, 4])
    # color_ls = sns.color_palette("husl", len(result_table))

    for ind in mean_table.index:
        label = "{}, AUPRC={:.4f}".format(name_converter[mean_table.loc[ind, 'model']], mean_table.loc[ind, 'prc'])
        plt.plot(mean_table.loc[ind]['rec'],
                 mean_table.loc[ind]['prec'],
                 # color=color_ls[ind],
                 label=label,
                 # linewidth=1
                 )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Sensitivity", fontweight='bold')
    plt.ylabel("Precision", fontweight='bold')

    # plt.title('Precision-Recall Curve (PRC) Analysis')
    plt.legend(loc='upper right')
    plt.savefig('../data/result/model_comparison/models_realtime_prc.pdf')
    plt.show()

    # plot ROC
    plt.figure(figsize=[5, 4])
    # color_ls = sns.color_palette("coolwarm_r", len(result_table))

    for ind in mean_table.index:
        label = "{}, AUROC={:.4f}".format(name_converter[mean_table.loc[ind, 'model']], mean_table.loc[ind, 'roc'])
        plt.plot(mean_table.loc[ind]['fpr'],
                 mean_table.loc[ind]['tpr'],
                 # color=color_ls[ind],
                 label=label,
                 # linewidth=1
                 )

    plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=0.5)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Flase Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')

    # plt.title('Receiver Operating Characteristic (ROC) Analysis')
    plt.legend(loc='lower right')
    plt.savefig('../data/result/model_comparison/models_realtime_roc.pdf')
    plt.show()

    # # plot sensitivity-alarm
    # plt.figure(figsize=[5, 4])
    # # color_ls = sns.color_palette("coolwarm_r", len(result_table))
    #
    # for ind in mean_table.index:
    #     prec = mean_table.loc[ind]['prec']
    #     rec = mean_table.loc[ind]['rec']
    #     alarm_rate = mean_table.loc[ind]['pos_rate'] * rec / prec * 60
    #     sens = rec[np.isclose(alarm_rate, 0.1, atol=0.01)][0]
    #     spec = 1 - np.interp(sens, mean_table.loc[ind, 'tpr'], mean_table.loc[ind, 'fpr'])
    #     plt.plot(alarm_rate,
    #              rec,
    #              # color=color_ls[ind],
    #              label="{}, Sens={:.4f}".format(name_converter[mean_table.loc[ind, 'model']], sens),
    #              # label="{}".format(name_converter[mean_table.loc[ind, 'model']]),
    #              # linewidth=1
    #              )
    #
    # plt.plot(0.1 * np.ones(10),
    #          np.linspace(-0.05, 1.05, 10),
    #          linestyle=(0, (9, 10)),
    #          color='black',
    #          label='Alarm Rate=0.1',
    #          linewidth=0.8
    #          )
    #
    # plt.xlim([-0.02, 0.52])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel("Intraoperative Alarm Rate (# Alarms/hr)", fontweight='bold')
    # plt.ylabel("Sensitivity", fontweight='bold')
    #
    # # plt.title('Sensitivity vs Alarm Rate')
    # plt.legend(loc='lower right')
    # plt.savefig('../data/result/model_comparison/models_realtime_sens_alarm.pdf')
    # plt.show()

    # plot alarm-sensitivity
    plt.figure(figsize=[5, 4])
    # color_ls = sns.color_palette("coolwarm_r", len(result_table))

    for ind in mean_table.index:
        prec = mean_table.loc[ind]['prec']
        rec = mean_table.loc[ind]['rec']
        alarm_rate = mean_table.loc[ind]['pos_rate'] * rec / prec * 60
        ar = alarm_rate[np.isclose(rec, 0.8, atol=0.05)][0]
        # spec = 1 - np.interp(sens, mean_table.loc[ind, 'tpr'], mean_table.loc[ind, 'fpr'])
        plt.plot(
                 rec,
                 alarm_rate,
                 # color=color_ls[ind],
                 label="{}, alert rate={:.2f}".format(name_converter[mean_table.loc[ind, 'model']], ar),
                 # linewidth=1
                 )

    plt.plot(0.8 * np.ones(10),
             np.linspace(-0.05, 10.05, 10),
             linestyle=(0, (9, 10)),
             color='black',
             label='Sensitivity cut=0.8',
             linewidth=0.8
             )

    plt.ylim([-0.1, 2.1])
    plt.xlim([-0.05, 1.05])
    plt.ylabel("Alert Rate (# Alert/hr)", fontweight='bold')
    plt.xlabel("Sensitivity", fontweight='bold')

    # plt.title('Sensitivity vs Alarm Rate')
    plt.legend(loc='upper left')
    plt.savefig('../data/result/model_comparison/models_realtime_alarm_sens.pdf')
    plt.show()


if __name__ == '__main__':
    table = pd.read_pickle('../data/result/model_comparison/realtime_models_random.pkl')
    # table_gbm = pd.read_pickle('../data/result/model_comparison/realtime_gbtree_random.pkl')
    # table = table.append(table_gbm, ignore_index=True)
    table = table.sort_values(by=['model', 'random_state'], ignore_index=True)

    name_converter = {
        "LSTM": "DIA-LSTM",
        "CNN": "DIA-CNN",
        "CatBoostClassifier": "GBM",
        "LogisticRegression": "LR",
        "RandomForestClassifier": "RF",
        "KNeighborsClassifier": "kNN",
        "LinearSVC": "SVM"
    }

    lstm = pd.read_pickle('/storage1/lu/Active/Hanyang/hybrid-inference/data_hypoxemia/result/lstm_horizon.pkl')
    table = table.append(lstm[lstm.horizon == 5].iloc[:, 1:], ignore_index=True)
    table.loc[40, 'model'] = 'LSTM'
    table.loc[40, 'random_state'] = 0

    cnn = pd.read_pickle('/storage1/lu/Active/Hanyang/hybrid-inference/data_hypoxemia/result/result_cnn.pkl')
    table = table.append(cnn[1:], ignore_index=True)
    table.loc[41, 'model'] = 'CNN'
    table.loc[41, 'random_state'] = 0

    plot_model_curves(table, name_converter)


    # labels = ['GBM', 'SVM', 'LR', 'RF']
    # x = np.arange(len(labels))
    # y_bot = [0.24, 0.26, 0.35, 0.65]
    # y_dif = [0.6 - 0.24, 0.8 - 0.26, 0.95 - 0.35, 1.45 - 0.65]
    # color_ls = sns.color_palette("coolwarm_r", len(labels))
    #
    # barlist = plt.bar(x, y_dif, bottom=y_bot, width=0.35)
    # # for ind in range(len(labels)):
    # #     barlist[ind].set_color(color_ls[ind])
    # plt.bar(x, y_dif, bottom=y_bot, width=0.35)
    # plt.xlim([-0.8, 3.8])
    # plt.ylim([0, 1.8])
    # plt.xticks(np.arange(4), labels)
    # plt.ylabel('Alarm Rate (# alarms per case hour)')
    # plt.title('Range of Alarm Rate (Sensitivity=[0.6, 0.8])')
    # # plt.yscale('log')
    # plt.savefig('data/result/alarm_rate_bar.png')
    # plt.show()


