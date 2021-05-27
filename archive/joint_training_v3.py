import numpy as np
import keras
from keras import models, layers
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
import tempfile
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_lstm_data import TransitionDataGenerator, PredictorDataGenerator
from sklearn.model_selection import train_test_split
from utils.model_cnn_rnn import get_model_lstm_w_att, cnn_model
from utils.model_rnn import lstm_1, lstm_2, lstm_3
import pickle
import sys


class JointDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, timeSeries, static, labels, batch_size=1024, window_size=10, horizon=5, shuffle=False):
        # "shuffle" cannot be set as True here !!
        'Initialization'
        self.n_features = len(timeSeries.columns) - 3
        self.batch_size = batch_size
        self.timeSeries = timeSeries
        self.static_features = static
        self.labels = labels
        self.list_data_index = list(timeSeries['index'].values)
        self.horizon = horizon
        self.shuffle = shuffle
        self.on_epoch_end()
        self.sliding_window = window_size
        self.dim = (self.sliding_window, self.n_features)

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
        X = np.empty((len(list_IDs_temp), *self.dim))
        # X_future: (n_samples, *dim)
        y = np.empty((len(list_IDs_temp), self.sliding_window, 1))
        list_IDs_seed = list(range(max(list_IDs_temp[0] - self.sliding_window, 0), list_IDs_temp[0])) + list_IDs_temp
        # get df segment for this batch plus several rows before the first ID in this batch
        df_seed = self.timeSeries[self.timeSeries['index'].isin(list_IDs_seed)]
        df_seed_cut = df_seed.copy()
        df_seed_cut.loc[list(df_seed_cut.groupby(['pid']).tail(self.horizon).index), df_seed_cut.columns[3:]] = 0
        for j in range(self.sliding_window):
            slice_X = df_seed_cut.groupby(['pid']).shift(periods=- j, fill_value=0).reset_index(level=0, drop=True)
            X[:, j, :] = slice_X.iloc[len(df_seed) - len(list_IDs_temp):, 3:]
            slice_Y = df_seed.groupby(['pid']).shift(periods=- j - self.horizon, fill_value=0).reset_index(level=0, drop=True)
            y[:, j, 0] = slice_Y.iloc[len(df_seed) - len(list_IDs_temp):]['SpO2']

        if len(y) != len(X):
            print("Error!!! - Shapes of X and y do not match!")

        # y: (n_samples, 1)
        labels = self.labels.loc[self.labels['index'].isin(list_IDs_temp), 'label'].values

        return X, [X, y, keras.utils.to_categorical(labels, num_classes=2)]


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


def multitask_rnn(window_size, n_features):

    n_in = window_size

    # define encoder
    visible = layers.Input(shape=(n_in, n_features))
    masked = layers.Masking(mask_value=0.)(visible)
    encoder = layers.LSTM(128, activation='relu')(masked)
    # define reconstruction decoder
    decoder1 = layers.RepeatVector(n_in)(encoder)
    decoder1 = layers.LSTM(128, activation='relu', return_sequences=True)(decoder1)
    decoder1 = layers.TimeDistributed(Dense(n_features), name='decoder1_output')(decoder1)
    # define forecasting decoder
    pred_hidden = layers.RepeatVector(n_in)(encoder)
    pred_hidden = layers.LSTM(64, activation='relu', return_sequences=True)(pred_hidden)
    decoder2 = layers.TimeDistributed(Dense(1), name='decoder2_output')(pred_hidden)
    # define outcome predictor
    predictor = layers.LSTM(64, activation='relu')(pred_hidden)
    predictor = layers.Dense(64, activation='relu')(predictor)
    predictor = layers.Dense(2, activation='softmax', name='predictor_output')(predictor)

    # tie it together
    model = models.Model(inputs=visible, outputs=[decoder1, decoder2, predictor])
    model.summary()
    keras.utils.plot_model(model, show_shapes=True, to_file='multitask_rnn_v3.png')

    model.compile(optimizer='adam', loss={'decoder1_output': 'mse',
                                          'decoder2_output': 'mse',
                                          'predictor_output': 'categorical_crossentropy'},
                  loss_weights={'decoder1_output': args.weight,
                                'decoder2_output': 1 - args.weight,
                                'predictor_output': 1 - args.weight})
    # model.compile(optimizer='adam', loss='mse')

    model_predictor = models.Model(inputs=model.inputs, outputs=predictor)

    return model, model_predictor


def multitask_rnn_2(window_size, n_features):

    n_in = window_size

    # define encoder
    visible = layers.Input(shape=(n_in, n_features))
    masked = layers.Masking(mask_value=0.)(visible)
    encoder = layers.LSTM(128, activation='relu', return_sequences=True)(masked)
    encoder = layers.LSTM(128, activation='relu')(encoder)
    # define reconstruction decoder
    decoder1 = layers.RepeatVector(n_in)(encoder)
    decoder1 = layers.LSTM(128, activation='relu', return_sequences=True)(decoder1)
    decoder1 = layers.LSTM(128, activation='relu', return_sequences=True)(decoder1)
    decoder1 = layers.TimeDistributed(Dense(n_features), name='decoder1_output')(decoder1)
    # define forecasting decoder
    pred_hidden = layers.RepeatVector(n_in)(encoder)
    pred_hidden = layers.LSTM(128, activation='relu', return_sequences=True)(pred_hidden)
    decoder2 = layers.LSTM(64, activation='relu', return_sequences=True)(pred_hidden)
    decoder2 = layers.TimeDistributed(Dense(1), name='decoder2_output')(decoder2)
    # define outcome predictor
    predictor = layers.LSTM(64, activation='relu')(pred_hidden)
    predictor = layers.Dense(64, activation='relu')(predictor)
    predictor = layers.Dense(2, activation='softmax', name='predictor_output')(predictor)

    # tie it together
    model = models.Model(inputs=visible, outputs=[decoder1, decoder2, predictor])
    model.summary()
    keras.utils.plot_model(model, show_shapes=True, to_file='multitask_rnn_v3.png')

    model.compile(optimizer='adam', loss={'decoder1_output': 'mse',
                                          'decoder2_output': 'mse',
                                          'predictor_output': 'categorical_crossentropy'},
                  loss_weights={'decoder1_output': args.weight,
                                'decoder2_output': 1 - args.weight,
                                'predictor_output': 1 - args.weight})
    # model.compile(optimizer='adam', loss='mse')

    model_predictor = models.Model(inputs=model.inputs, outputs=predictor)

    return model, model_predictor


def train(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args):

    batch_size = args.batch_size
    window_size = args.steps
    n_features = len(timeSeriesTr.columns) - 3
    train_data_generator = JointDataGenerator(timeSeriesTr, static_feat, labelsTr,
                                              batch_size=batch_size,
                                              window_size=window_size,
                                              horizon=args.horizon,
                                              shuffle=False)
    valid_data_generator = JointDataGenerator(timeSeriesTe, static_feat, labelsTe,
                                              batch_size=batch_size,
                                              window_size=window_size,
                                              horizon=args.horizon,
                                              shuffle=False)

    '''Choose model for the prediction'''
    model, model_predictor = multitask_rnn(window_size, n_features)

    '''Fit model'''
    print(args)
    model.fit_generator(generator=train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=args.epochs,
                        use_multiprocessing=True,
                        verbose=1
                        )

    model_encoder = models.Model(inputs=model.inputs, outputs=model.layers[2].output)

    return model, model_encoder


def evaluate(joint_model, test_data_generator, pos_rate):
    # Testing
    pred_model = models.Model(inputs=joint_model.input, outputs=joint_model.layers[-1].output)
    y_prob = pred_model.predict_generator(test_data_generator)[:, 1]
    y_test = test_data_generator[0][1][:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)

    print('--------------------------------------------')
    print('Evaluation of test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
          "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))

    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)
    alarm_rate = pos_rate * sensitivity / PPV
    print("sensitivity:", "%0.4f" % sensitivity,
          "specificity:", "%0.4f" % specificity,
          "PPV:", "%0.4f" % PPV,
          "NPV:", "%0.4f" % NPV,
          "F1 score:", "%0.4f" % f1,
          "accuracy:", "%0.4f" % acc)
    print("Alarm rate:", alarm_rate)
    print('--------------------------------------------')

    spo2_model = models.Model(inputs=joint_model.input, outputs=joint_model.layers[-2].output)
    out = spo2_model.predict_generator(test_data_generator)
    y_pred = np.zeros(len(out), )

    for ind in range(len(out)):
        window = out[ind, 5:, 0]
        if_low = window <= 0.9
        if np.sum(if_low) == 5:
            y_pred[ind] = 1

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

    print("sensitivity:", "%0.4f" % sensitivity,
          "specificity:", "%0.4f" % specificity,
          "PPV:", "%0.4f" % PPV,
          "NPV:", "%0.4f" % NPV,
          "F1 score:", "%0.4f" % f1,
          "accuracy:", "%0.4f" % acc)
    print("Alarm rate:", alarm_rate)
    print('--------------------------------------------')

    # result_table = pd.DataFrame(columns=['model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])
    # result_table = result_table.append({
    #     'model': 'LSTM',
    #     'fpr': fpr,
    #     'tpr': tpr,
    #     'roc': metrics.auc(fpr, tpr),
    #     'prec': prec,
    #     'rec': rec,
    #     'prc': metrics.auc(rec, prec),
    #     'y_test': y_test,
    #     'y_prob': y_prob,
    #     'pos_rate': pos_rate
    # }, ignore_index=True)
    #
    # # save results
    # result_table.to_pickle('data/result/realtime_lstm.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=5)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--batch_size', type=int, default=pow(2, 14))
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--weight', type=float, default=0.1)
    parser.add_argument('--model_path', type=str, default="data/model/model_joint_v3.h5")
    args = parser.parse_args()
    print(args)

    df_static = pd.read_csv(config.get('processed', 'df_static_file'))
    df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
    static_feat = pd.read_csv('../data/features/static-notxt.csv')

    timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate = prepare_data(df_static, df_dynamic)
    model, _ = train(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args)
    model.save(args.model_path)
    joint_model = models.load_model(args.model_path)
    test_data_generator = PredictorDataGenerator(timeSeriesTe, static_feat, labelsTe,
                                                 batch_size=len(labelsTe),
                                                 window_size=args.steps,
                                                 n_classes=2)
    evaluate(joint_model, test_data_generator, pos_rate)









