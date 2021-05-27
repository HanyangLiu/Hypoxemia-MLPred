import numpy as np
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from keras.callbacks import ModelCheckpoint
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric
from keras.models import model_from_json
import argparse
import pandas as pd
from file_config.config import config
from sklearn import preprocessing
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
from utils.utility_lstm_data import DataGenerator
from utils.model_cnn_rnn import get_model_lstm_w_att, cnn_model
from utils.model_rnn import lstm_1, lstm_2, lstm_3
import pickle
import sys


def prepare_data(df_static, df_dynamic):
    '''Prepare Data'''
    # label assignment (according to imputed SpO2)
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)

    path_sta_label = 'data/label/static_label_lstm_' + str(args.hypoxemia_window) + '.pkl'
    path_dyn_label = 'data/label/dynamic_label_lstm_' + str(args.hypoxemia_window) + '.pkl'
    label_assign = LabelAssignment(hypoxemia_thresh=args.hypoxemia_thresh,
                                   hypoxemia_window=args.hypoxemia_window,
                                   prediction_window=args.prediction_window)
    # print('Assigning labels...')
    # static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
    # static_label.to_pickle(path_sta_label)
    # dynamic_label.to_pickle(path_dyn_label)

    static_label = pd.read_pickle(path_sta_label)
    dynamic_label = pd.read_pickle(path_dyn_label)
    positive_pids = label_assign.get_positive_pids(static_label)
    print('Done.')

    # normalization of data
    min_max_scaler = preprocessing.MinMaxScaler()
    data = df_dynamic.iloc[:, 3:].values
    df_dynamic.iloc[:, 3:] = min_max_scaler.fit_transform(data)

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
                                                 test_size=0.2,
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
    to_keep = (dynamic_label['if_to_drop'] == 0).values
    is_in_train = dynamic_label[['pid']].isin(pid_train)['pid'].values
    is_in_test = dynamic_label[['pid']].isin(pid_test)['pid'].values
    # dynamic_label.loc[list(dynamic_label[dynamic_label.if_to_drop == 1].index), 'label'] = 2
    selected_idx_train = list(np.where(to_keep & is_in_train)[0])
    selected_idx_test = list(np.where(to_keep & is_in_test)[0])

    timeSeriesTr = df_dynamic.iloc[selected_idx_train, 0:21]
    labelsTr = dynamic_label.iloc[selected_idx_train][['index', 'label']]
    timeSeriesTe = df_dynamic.iloc[selected_idx_test, 0:21]
    labelsTe = dynamic_label.iloc[selected_idx_test][['index', 'label']]

    num_pos = np.sum(labelsTr['label'].values) + np.sum(labelsTe['label'].values)
    num_all = len(labelsTr) + len(labelsTe)
    pos_rate = num_pos / num_all
    print('Positive samples:', num_pos, '/', num_all)
    print('Ratio:', '%0.2f' % (num_pos/num_all*100), '%')

    return timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate


def train_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args):

    batch_size = args.batch_size
    window_size = args.steps
    n_features = len(timeSeriesTr.columns) + len(static_feat.columns) - 4
    train_data_generator = DataGenerator(timeSeriesTr, static_feat, labelsTr,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2,
                                         shuffle=False)
    valid_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2,
                                         shuffle=False)

    '''Choose model for the prediction'''
    model = lstm_1(window_size, n_features, n_classes=2)

    # define the checkpoint
    model_path = args.model_path
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit_generator(generator=train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=args.epochs,
                        use_multiprocessing=True,
                        verbose=1,
                        callbacks=callbacks_list
                        )

    return model


def evaluate(model, test_data_generator, pos_rate):
    # Testing
    y_prob = model.predict_generator(test_data_generator)[:, 1]
    y_test = test_data_generator[0][1][:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)
    alarm_rate = pos_rate * sensitivity / PPV

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
    print("Alarm rate:", alarm_rate)
    print('--------------------------------------------')

    result_table = pd.DataFrame(columns=['model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])
    result_table = result_table.append({
        'model': 'LSTM',
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

    # save results
    result_table.to_pickle('data/result/realtime_lstm.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--batch_size', type=int, default=pow(2, 14))
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, default="data/model/model_lstm.h5")
    args = parser.parse_args()
    print(args)

    df_static = pd.read_csv(config.get('processed', 'df_static_file'))
    df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
    static_feat = pd.read_csv('data/features/static-notxt.csv')

    timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate = prepare_data(df_static, df_dynamic)
    model = train_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args)
    test_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                        batch_size=len(labelsTe),
                                        window_size=args.steps,
                                        n_classes=2)
    evaluate(model, test_data_generator, pos_rate)



