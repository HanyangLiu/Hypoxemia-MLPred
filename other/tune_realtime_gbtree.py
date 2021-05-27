
from train_realtime_gbtree import prepare_data, train_gbtree, evaluate
from utils.utility_analysis import line_search_best_metric, count_correct_label, best_ntree_score
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
parser.add_argument('--feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
parser.add_argument('--random_state', type=int, default=1)
parser.add_argument('--gb_tool', type=str, default='catboost')
parser.add_argument('--if_tuning', type=str, default='False')
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()

df_static = pd.read_csv(config.get('processed', 'df_static_file'))
df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
dynamic_feature = pd.read_csv('data/features/' + args.feature_file)

X_train, X_test, y_train, y_test, pos_rate = prepare_data(df_static,
                                                          df_dynamic,
                                                          dynamic_feature,
                                                          args=args)

AU_PRC = []
best_auprc = 0
best_param = [0, 0, 0]
for lr in [0.02, 0.05, 0.1, 0.3]:
    for depth in [1, 3, 6]:
        for l2 in [1, 3]:
            args.lr = lr
            args.depth = depth
            args.l2 = l2
            print(args)

            model = train_gbtree(X_train, y_train, pos_rate, args)
            print('Params:', lr, depth, l2)
            auprc = evaluate(model, X_test, y_test, pos_rate, args)
            AU_PRC.append(auprc)

            if auprc > best_auprc:
                best_auprc = auprc
                best_param = [lr, depth, l2]

print('Best params:', best_param)
print('Best AU-PRC:', best_auprc)
print(AU_PRC)












