import numpy as np
from numpy import array
from numpy import hstack
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric
import argparse
import pandas as pd
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
from utils.utility_training import DataGenerator
from utils.model_cnn_rnn import get_model_lstm_w_att, cnn_model
from utils.model_rnn import lstm_1, lstm_2
import pickle
import sys


def evaluate(model, test_data_generator):
    # Testing
    y_prob = model.predict_generator(test_data_generator)[:, 1]
    y_test = test_data_generator[0][1][:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

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

    # plot ROC and PRC
    plot_roc(fpr, tpr, 'data/result/roc_initial.png')
    plot_prc(rec, prec, 'data/result/pr_initial.png')


parser = argparse.ArgumentParser(description='hypoxemia prediction')
parser.add_argument('--hypoxemia_thresh', type=int, default=90)
parser.add_argument('--hypoxemia_window', type=int, default=10)
parser.add_argument('--prediction_window', type=int, default=5)
parser.add_argument('--filter_mode', type=str, default='exclude')
args = parser.parse_args()
print(args)


df_static = pd.read_csv(config.get('processed', 'df_static_file'))
df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
static_feat = pd.read_csv('data/features/static-notxt.csv')


'''Prepare Data'''
# label assignment (according to imputed SpO2)
print('Assigning labels...')
imputer = DataImputation()
df_static = imputer.impute_static_dataframe(df_static)
df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
label_assign = LabelAssignment(hypoxemia_thresh=args.hypoxemia_thresh,
                               hypoxemia_window=args.hypoxemia_window,
                               prediction_window=args.prediction_window)
static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
positive_pids = label_assign.get_positive_pids(static_label)
print('Done.')

# get subgroup pids
subgroup_pids = PatientFilter(df_static=df_static,
                              mode=args.filter_mode,
                              include_icd=['J96.', 'J98.', '519.', '518.', '277.0', 'E84', 'Q31.5', '770.7',
                                           'P27.1', '490', '491', '492', '493', '494', '495', '496', 'P27.8',
                                           'P27.9', 'J44', 'V46.1', 'Z99.1'],  # High-risk group
                              exclude_icd9=['745', '746', '747'],
                              exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()

# split subgroup pids into training and test pid set
pid_train, pid_test, _, _ = train_test_split(static_label.loc[subgroup_pids]['pid'].values,
                                             static_label.loc[subgroup_pids]['label'].values,
                                             test_size=0.1,
                                             random_state=0,
                                             stratify=static_label.loc[subgroup_pids]['label'].values)
pid_train = sorted(list(pid_train))
pid_test = sorted(list(pid_test))

print('Positive Patient:', len(set(subgroup_pids) & set(positive_pids)), '/', len(subgroup_pids))
print('Before trimming:', len(positive_pids), '/', len(df_static))
print('Trimmed cases:', len(df_static) - len(subgroup_pids))

# select feature rows with pid in subgroup as data matrix
print('Training/testing split:', len(pid_train), '/', len(pid_test))
print('Split into training and test set...')
is_in_train = dynamic_label[['pid']].isin(pid_train)['pid'].values
is_in_test = dynamic_label[['pid']].isin(pid_test)['pid'].values
selected_idx_train = list(np.where(is_in_train)[0])
selected_idx_test = list(np.where(is_in_test)[0])

timeSeriesTr = df_dynamic.iloc[selected_idx_train, 0:15]
labelsTr = dynamic_label.iloc[selected_idx_train][['index', 'label']]
timeSeriesTe = df_dynamic.iloc[selected_idx_test, 0:15]
labelsTe = dynamic_label.iloc[selected_idx_test][['index', 'label']]

batch_size = pow(2, 14)
window_size = 10
n_features = len(timeSeriesTr.columns) + len(static_feat.columns) - 3
train_data_generator = DataGenerator(timeSeriesTr, static_feat, labelsTr,
                                     batch_size=batch_size,
                                     window_size=window_size)
valid_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                     batch_size=batch_size,
                                     window_size=window_size)
test_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                    batch_size=len(timeSeriesTe),
                                    window_size=window_size)

model = lstm_2(window_size, n_features)
# model = cnn_model()
# model.summary()

model.fit_generator(generator=train_data_generator,
                    validation_data=valid_data_generator,
                    epochs=10,
                    use_multiprocessing=True,
                    verbose=1
                    )
pickle.dump(model, open('data/model/realtime_lstm.sav', 'wb'))
model = pickle.load(open('data/model/realtime_lstm.sav', 'rb'))

evaluate(model, test_data_generator)


