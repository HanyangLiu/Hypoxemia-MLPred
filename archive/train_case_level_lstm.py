import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, Convolution1D
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric
import argparse
import pandas as pd
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
from utils.model_cnn_rnn import get_model_lstm_w_att, cnn_model
from utils.model_rnn import lstm_1, lstm_2, lstm_3
import pickle
import sys


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, timeSeries, static, labels, dynamic_label, batch_size=128, window_size=100, n_classes=2, shuffle=False):
        'Initialization'
        self.n_features = len(timeSeries.columns) + len(static.columns) - 4
        self.batch_size = batch_size
        self.timeSeries = timeSeries
        self.static_features = static
        self.labels = labels
        self.hypoxemia_time = dynamic_label[dynamic_label.label == 1].groupby(['pid']).min()['ts']
        self.list_data_index = list(timeSeries['pid'].unique())
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.observation_window = window_size
        self.dim = (self.observation_window, self.n_features)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_data_index) / self.batch_size))

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_data_index[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_data_index))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # X: (n_samples, *dim)
        X = np.zeros((len(list_IDs_temp), *self.dim))
        for idx, pid in enumerate(list_IDs_temp):
            df = self.timeSeries[self.timeSeries.pid == pid]
            if self.labels.loc[pid, 'label'] == 1:
                df = df[df['ts'] < self.hypoxemia_time[pid]]
            slice = pd.merge(df, self.static_features, how='left', on='pid').iloc[:, 3:].values
            if slice.shape[0] < self.observation_window:
                # X[idx, self.observation_window - slice.shape[0]:, :] = slice
                X[idx, 0:slice.shape[0], :] = slice
            else:
                X[idx, :, :] = slice[slice.shape[0] - self.observation_window, :]
        y = self.labels[self.labels['pid'].isin(list_IDs_temp)]['label'].values

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def prepare_data(df_static, df_dynamic):
    '''Prepare Data'''
    # label assignment (according to imputed SpO2)
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)

    path_sta_label = 'data/label/static_label_lstm_' + str(args.hypoxemia_window) + '.pkl'
    path_dyn_label = 'data/label/dynamic_label_lstm_' + str(args.hypoxemia_window) + '.pkl'
    label_assign = LabelAssignment(hypoxemia_thresh=90,
                                   hypoxemia_window=args.hypoxemia_window,
                                   prediction_window=5)
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
                                  mode='exclude',
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
    is_in_train = dynamic_label[['pid']].isin(pid_train)['pid'].values
    is_in_test = dynamic_label[['pid']].isin(pid_test)['pid'].values
    # dynamic_label.loc[list(dynamic_label[dynamic_label.if_to_drop == 1].index), 'label'] = 2
    selected_idx_train = list(np.where(is_in_train)[0])
    selected_idx_test = list(np.where(is_in_test)[0])

    timeSeriesTr = df_dynamic.iloc[selected_idx_train, 0:21]
    labelsTr = static_label.iloc[pid_train][['pid', 'label']]
    timeSeriesTe = df_dynamic.iloc[selected_idx_test, 0:21]
    labelsTe = static_label.iloc[pid_test][['pid', 'label']]

    num_pos = np.sum(labelsTr['label'].values) + np.sum(labelsTe['label'].values)
    num_all = len(labelsTr) + len(labelsTe)
    pos_rate = num_pos / num_all

    return timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate, dynamic_label


def lstm(window_size, n_features, n_classes):

    model = Sequential()
    model.add(
        keras.layers.GRU(128, activation='relu', recurrent_dropout=0.3, input_shape=(window_size, n_features), return_sequences=True))
    model.add(keras.layers.GRU(128, activation='relu', unroll=True))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    return model


def train_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, dynamic_label, args):

    batch_size = args.batch_size
    window_size = args.steps
    n_features = len(timeSeriesTr.columns) + len(static_feat.columns) - 4
    train_data_generator = DataGenerator(timeSeriesTr, static_feat, labelsTr, dynamic_label,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2)
    valid_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe, dynamic_label,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2)

    '''Choose model for the prediction'''
    model = lstm(window_size, n_features, n_classes=2)
    # model = keras.models.load_model(args.model_path)
    # model = cnn_model()
    # model.summary()

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
    result_table.to_pickle('data/result/caselevel_lstm.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="data/model/model_case_lstm.h5")
    args = parser.parse_args()
    print(args)

    df_static = pd.read_csv('../data/data_frame/static_dataframe.csv')
    df_dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')
    static_feat = pd.read_csv('../data/features/static-notxt.csv')

    timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate, dynamic_label = prepare_data(df_static, df_dynamic)
    model = train_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, dynamic_label, args)
    # model = keras.models.load_model(args.model_path)
    test_data_generator = DataGenerator(timeSeriesTe, static_feat, labelsTe, dynamic_label,
                                        batch_size=14700,
                                        window_size=args.steps,
                                        n_classes=2)
    evaluate(model, test_data_generator, pos_rate)


