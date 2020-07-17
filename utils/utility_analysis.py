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
    plt.savefig(save_dir)
    plt.show()


def plot_prc(rec, prec, save_dir):

    plt.figure()
    plt.title('Precision Recall Curve')
    plt.plot(rec, prec, 'b', label='AUC = %0.2f' % metrics.auc(rec, prec))
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(save_dir)
    plt.show()


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


def au_prc(y_true, y_prob):

    prec, rec, _ = metrics.precision_recall_curve(y_true, y_prob)
    au_prc = metrics.auc(rec, prec)

    return au_prc


def line_search_best_metric(y_test, y_prob, spec_thresh=0.95):
    """
    Line search the best threshold to balance the trade-off between sens/spec
    :param y_test:
    :param y_prob:
    :param spec_thresh:
    :return:
    """

    t = np.arange(0.0, 1.0, 0.01)
    diff = 1.0

    for i in range(t.shape[0]):
        dt = t[i] - 0.5
        sens, spec, PPV, NPV, f1, acc = metric_eval(y_test, np.round(y_prob - dt))
        if abs(spec - spec_thresh) < diff:
            best_t = t[i]
            best_metrics = (sens, spec, PPV, NPV, f1, acc)
            y_pred = np.round(y_prob - dt)
            diff = abs(spec - 0.95)

    return best_metrics, y_pred


def best_ntree_score(estimator, X):
    """
    This scorer uses the best_ntree_limit to return
    the best y_prob
    """
    try:
        y_prob = estimator.predict_proba(X, ntree_limit=estimator.best_ntree_limit)
    except AttributeError:
        y_prob = estimator.predict_proba(X)[:, 1]
    return y_prob


def count_correct_label(y_test, y_prob, win_size):
    """
    Analyze the correctly predicted label distribution
    :param y_test:
    :param y_pred:
    :param win_size:
    :return:
    """
    _, y_pred = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

    idx_arr = np.where(y_test)[0]
    horizon_arr = np.zeros(len(y_pred))
    for i, y_idx in enumerate(idx_arr):
        if i == len(idx_arr) - 1:
            continue
        if idx_arr[i + 1] != idx_arr[i] + 1:
            try:
                horizon_arr[y_idx - win_size + 1: y_idx + 1] = np.array(list(np.linspace(win_size, 1, win_size)))
            except:
                continue

    counts_gt = []
    for j in list(np.linspace(1, win_size, win_size)):
        ids = np.where(horizon_arr == j)
        count = np.sum([y_test[i] == 1 for i in ids])
        counts_gt.append(count)
    print('Groundtruth:', counts_gt)

    counts = []
    for j in list(np.linspace(1, win_size, win_size)):
        ids = np.where(horizon_arr == j)
        count = np.sum([y_pred[i] == 1 for i in ids])
        counts.append(count)
    print('Result:', counts)

    return counts_gt, counts


def first_correct_label(y_test, y_prob, win_size):
    """
    Analyze the first correctly predicted label distribution
    :param y_test:
    :param y_pred:
    :param win_size:
    :return:
    """
    _, y_pred = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

    idx_arr = np.where(y_test)[0]
    horizon_arr = np.zeros(len(y_pred))
    first_correct = []
    for i, y_idx in enumerate(idx_arr):
        if i == len(idx_arr) - 1:
            continue
        if idx_arr[i + 1] != idx_arr[i] + 1:
            try:
                horizon_arr[y_idx - win_size + 1: y_idx + 1] = np.array(list(np.linspace(win_size, 1, win_size)))
                correct = y_pred[y_idx - win_size + 1: y_idx + 1] == y_test[y_idx - win_size + 1: y_idx + 1]
                first_correct.append(np.where(correct)[0][0])
            except:
                continue
    x = [str(int(i)) for i in np.linspace(win_size, 1, win_size)]
    x_pos = [i for i, _ in enumerate(x)]
    print([sum(np.array(first_correct) == i) for i in range(win_size)])
    plt.figure()
    plt.bar(x_pos, [sum(np.array(first_correct) == i) for i in range(win_size)])
    plt.xticks(x_pos, x)
    plt.title('win_size =' + str(win_size))
    plt.savefig('win_size' + str(win_size) + '.png')
    plt.close()

    return first_correct


def estimate_prediction_horizon(y_test, y_prob, win_size):
    """
    Analyze the first correctly predicted label distribution
    :param y_test:
    :param y_pred:
    :param win_size:
    :return:
    """
    _, y_pred = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

    idx_arr = np.where(y_test)[0]
    horizon_arr = np.zeros(len(y_pred))
    for i, y_idx in enumerate(idx_arr):
        if i == len(idx_arr) - 1:
            continue
        if idx_arr[i + 1] != idx_arr[i] + 1:
            try:
                horizon_arr[y_idx - win_size + 1: y_idx + 1] = np.array(list(np.linspace(win_size, 1, win_size)))
            except:
                continue

    counts = []
    for j in list(np.linspace(1, win_size, win_size)):
        j = int(j)
        ids = np.where(horizon_arr == j)[0]
        count = np.sum([np.sum(y_pred[i: i+j]) == j for i in ids])
        counts.append(count)
    print('Result:', counts)

    x = [str(int(i)) for i in np.linspace(win_size, 1, win_size)]
    x_pos = [i for i, _ in enumerate(x)]
    plt.figure()
    plt.bar(x_pos, counts)
    plt.xticks(x_pos, x)
    plt.title('Correct predicted time horizon: win_size =' + str(win_size))
    plt.xlabel('Minutes to hypoxemia')
    plt.ylabel('Frequency')
    plt.savefig('data/result/win_size' + str(win_size) + '.png')
    plt.close()

    return counts



