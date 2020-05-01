
from train_realtime_predictor import prepare_data, train_gbtree
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric, count_correct_label
import matplotlib.pyplot as plt
from file_config.config import config
from sklearn import metrics
import pandas as pd
import numpy as np
import argparse

# initialize arguments
parser = argparse.ArgumentParser(description='hypoxemia prediction')
parser.add_argument('--hypoxemia_thresh', type=int, default=90)
parser.add_argument('--hypoxemia_window', type=int, default=10)
parser.add_argument('--prediction_window', type=int, default=5)
parser.add_argument('--filter_mode', type=str, default='exclude')
parser.add_argument('--dynamic_feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
args = parser.parse_args()

df_static = pd.read_csv(config.get('processed', 'df_static_file'))
df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
dynamic_feature = pd.read_csv('data/features/' + args.dynamic_feature_file)

FPR, TPR, PREC, REC = [], [], [], []
METRIC = []

for win_size in [15]:

    args.prediction_window = win_size
    print(args)

    X_train, X_test, y_train, y_test = prepare_data(df_static,
                                                    df_dynamic,
                                                    dynamic_feature,
                                                    args=args)
    model = train_gbtree(X_train, y_train)

    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

    FPR.append(fpr)
    TPR.append(tpr)
    PREC.append(PREC)
    REC.append(REC)
    METRIC.append([sensitivity, specificity, PPV, NPV, f1, acc])

    print('--------------------------------------------')
    print('Evaluation of test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
          "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))
    print("sensitivity:", "%0.4f" % sensitivity,
          "specificity:", "%0.4f" % specificity,
          "PPV:", "%0.4f" % PPV,
          "NPV:", "%0.4f" % NPV,
          "F1 score:", "%0.4f" % f1,
          "accuracy:", "%0.4f" % acc)
    print('--------------------------------------------')

    # analyze the correctly predicted label distribution
    count_correct_label(y_test, y_prob, win_size)







