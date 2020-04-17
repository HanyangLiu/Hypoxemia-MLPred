import os
import pandas as pd
import numpy as np
from file_config.config import config
import matplotlib.pyplot as plt
from sklearn import metrics


def feature_value_filling_ratio(df):
    ratios = []
    for column in list(df.columns):
        a = df[df[column].isnull()]
        ratio = 1 - len(a) / len(df)
        ratios.append(ratio)

    filling_ratio = dict(zip(list(df.columns), ratios))

    return filling_ratio


def plot_roc(fpr, tpr, save_dir):

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % metrics.auc(fpr, tpr))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig(save_dir)


def plot_prc(rec, prec, save_dir):

    plt.figure()
    plt.title('Precision Recall Curve')
    plt.plot(rec, prec, 'b', label='AUC = %0.2f' % metrics.auc(rec, prec))
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.savefig(save_dir)


def metric_eval(y_test, y_pred):

    C = metrics.confusion_matrix(y_test, y_pred)
    tn = np.float(C[0][0])
    fn = np.float(C[1][0])
    tp = np.float(C[1][1])
    fp = np.float(C[0][1])

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    PPV = tp / (tp + fp) if (tp + fp) != 0 else 0
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0
    f1 = metrics.f1_score(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)

    return sensitivity, specificity, PPV, NPV, f1, acc


def line_search_best_metric(y_test, y_prob, spec_thresh):
    '''
    Line search the best threshold to balance the trade-off between sens/spec
    :param y_test:
    :param y_prob:
    :param spec_thresh:
    :return:
    '''

    t = np.arange(0.0, 1.0, 0.01)
    diff = 1.0

    for i in range(t.shape[0]):
        dt = t[i] - 0.5
        sens, spec, PPV, NPV, f1, acc = metric_eval(y_test, np.round(y_prob - dt))
        if abs(spec - spec_thresh) < diff:
            best_t = t[i]
            best_metrics = (sens, spec, PPV, NPV, f1, acc)
            diff = abs(spec - 0.95)

    return best_metrics

