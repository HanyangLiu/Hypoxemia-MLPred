
from train_realtime_gbtree import prepare_data, train_gbtree
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric, estimate_prediction_horizon, first_correct_label
import matplotlib.pyplot as plt
import seaborn as sns
from file_config.config import config
from sklearn import metrics
import pandas as pd
from ast import literal_eval
import numpy as np
import argparse
import os


df_static = pd.read_csv('data/data_frame/static_dataframe.csv')
df_dynamic = pd.read_csv('data/data_frame/dynamic_dataframe.csv')
dynamic_feature = pd.read_csv('data/features/' + 'dynamic-ewm-notxt-nonimp.csv')

# initialize arguments
parser = argparse.ArgumentParser(description='hypoxemia prediction')
parser.add_argument('--hypoxemia_thresh', type=int, default=90)
parser.add_argument('--hypoxemia_window', type=int, default=10)
parser.add_argument('--prediction_window', type=int, default=5)
parser.add_argument('--filter_mode', type=str, default='exclude')
parser.add_argument('--dynamic_feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
parser.add_argument('--random_state', type=int, default=1)
parser.add_argument('--gb_tool', type=str, default='xgboost')
parser.add_argument('--if_tuning', type=str, default='False')
parser.add_argument('--n_jobs', type=int, default=-1)

parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--l2', type=int, default=3)
args = parser.parse_args()


result_table = pd.DataFrame(columns=['window', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])

for win_size in [1, 2, 3, 4, 5, 10, 15, 20, 25]:

    args.prediction_window = win_size
    print(args)

    X_train, X_test, y_train, y_test, pos_rate = prepare_data(df_static,
                                                    df_dynamic,
                                                    dynamic_feature,
                                                    args=args)
    model = train_gbtree(X_train, y_train, pos_rate, args=args)

    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

    result_table = result_table.append({
        'window': win_size,
        'fpr': fpr,
        'tpr': tpr,
        'roc': metrics.auc(fpr, tpr),
        'prec': prec,
        'rec': rec,
        'prc': metrics.auc(rec, prec),
        'y_test': y_test,
        'y_prob': y_prob,
        'pos_rate': pos_rate
    }, ignore_index=True)

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
    # estimate_prediction_horizon(y_test, y_prob, win_size)

# Set window size as index labels
result_table.set_index('window', inplace=True)
# save results
result_table.to_pickle('data/result/various_pred_win.pkl')









