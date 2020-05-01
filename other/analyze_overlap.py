import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from train_realtime_predictor import prepare_data, train_gbtree
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import sys

df_static = pd.read_csv(config.get('processed', 'df_static_file'))
df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))

INDs = []
for hypoxemia_window in np.linspace(1, 10, 10).astype(int):
    print(hypoxemia_window)
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
    label_assign = LabelAssignment(hypoxemia_thresh=90,
                                   hypoxemia_window=hypoxemia_window,
                                   prediction_window=5)
    static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)

    labels = dynamic_label['label'].values
    INDs.append(list(np.where(labels == 1)))


# initialize arguments
parser = argparse.ArgumentParser(description='hypoxemia prediction')
parser.add_argument('--hypoxemia_thresh', type=int, default=90)
parser.add_argument('--hypoxemia_window', type=int, default=1)
parser.add_argument('--prediction_window', type=int, default=5)
parser.add_argument('--filter_mode', type=str, default='exclude')
parser.add_argument('--dynamic_feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
args = parser.parse_args()


dynamic_feature = pd.read_csv('data/features/' + args.dynamic_feature_file)


print(args)

X_train, X_test, y_train, y_test = prepare_data(df_static,
                                                df_dynamic,
                                                dynamic_feature,
                                                args=args)
model = train_gbtree(X_train, y_train)
X = dynamic_feature.iloc[:, 2:].values

# Testing
y_prob = model.predict_proba(X)[:, 1]

probs = []
for inds in INDs:
    probs.append(np.mean(y_prob[inds]))

print(probs)






