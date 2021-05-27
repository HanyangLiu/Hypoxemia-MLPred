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
from utils.utility_lstm_data import SpO2DataGenerator, PredictorDataGenerator
from sklearn.metrics import mean_squared_error
import argparse
import pandas as pd
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
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


def rnn_autoencoder(window_size, n_features):

    n_in = window_size
    n_out = window_size

    # define encoder
    visible = layers.Input(shape=(n_in, n_features))
    masked = layers.Masking(mask_value=0.)(visible)
    encoder = layers.LSTM(128, activation='relu')(masked)
    # define reconstruction decoder
    decoder1 = layers.RepeatVector(n_in)(encoder)
    decoder1 = layers.LSTM(128, activation='relu', return_sequences=True)(decoder1)
    decoder1 = layers.TimeDistributed(Dense(n_features))(decoder1)
    # define prediction decoder
    decoder2 = layers.RepeatVector(n_out)(encoder)
    decoder2 = layers.LSTM(128, activation='relu', return_sequences=True)(decoder2)
    decoder2 = layers.TimeDistributed(Dense(1))(decoder2)
    # tie it together
    model = models.Model(inputs=visible, outputs=[decoder1, decoder2])
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    try:
        keras.utils.plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
    except:
        print('>>>> plot not working!')

    return model


def predictor(model_encoder, args):


    n_in = args.steps
    # model_encoder = models.load_model(args.encoder_path)
    dense = layers.Dense(128, activation='relu')(model_encoder.layers[-1].output)
    dense = layers.Dropout(0.3)(dense)
    out = layers.Dense(2, activation='softmax')(dense)

    model = models.Model(inputs=model_encoder.layers[0].input, outputs=out)
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )
    keras.utils.plot_model(model, show_shapes=True, to_file='predictor.png')

    return model


def transition_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args):

    batch_size = args.batch_size
    window_size = args.steps
    n_features = len(timeSeriesTr.columns) - 3
    train_data_generator = SpO2DataGenerator(timeSeriesTr, static_feat, labelsTr,
                                                   batch_size=batch_size,
                                                   window_size=window_size,
                                                   horizon=args.horizon,
                                                   shuffle=False)
    valid_data_generator = SpO2DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                                   batch_size=batch_size,
                                                   window_size=window_size,
                                                   horizon=args.horizon,
                                                   shuffle=False)

    '''Choose model for the prediction'''
    model = rnn_autoencoder(window_size, n_features)

    '''Fit model'''
    model.fit_generator(generator=train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=args.epochs_1,
                        use_multiprocessing=True,
                        verbose=1
                        )

    model_encoder = models.Model(inputs=model.inputs, outputs=model.layers[2].output)

    return model, model_encoder


def prediction_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, model_encoder, args):

    batch_size = args.batch_size
    window_size = args.steps
    n_features = len(timeSeriesTr.columns) + len(static_feat.columns) - 3
    train_data_generator = PredictorDataGenerator(timeSeriesTr, static_feat, labelsTr,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2)
    valid_data_generator = PredictorDataGenerator(timeSeriesTe, static_feat, labelsTe,
                                         batch_size=batch_size,
                                         window_size=window_size,
                                         n_classes=2)

    model = predictor(model_encoder, args)

    model.fit_generator(generator=train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=args.epochs_2,
                        use_multiprocessing=True,
                        verbose=1
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


def evaluate_generator(model, test_data_generator, spo2_data_generator):

    spo2_model = models.Model(inputs=model.input, outputs=model.layers[-1].output)
    y_test = test_data_generator[0][1][:, 1]
    spo2_test = spo2_data_generator[0][1]
    spo2_pred = spo2_model.predict_generator(spo2_data_generator)
    rmse_spo2 = mean_squared_error(spo2_test.reshape(len(spo2_test) * args.steps), spo2_pred.reshape(len(spo2_pred) * args.steps))
    y_pred = np.zeros(len(spo2_pred), )

    for ind in range(len(spo2_pred)):
        window = spo2_pred[ind, 5:, 0]
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
    print('--------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=5)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--batch_size', type=int, default=pow(2, 14))
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--epochs_1', type=int, default=10)
    parser.add_argument('--epochs_2', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--encoder_path', type=str, default="data/model/model_encoder.h5")
    parser.add_argument('--model_path', type=str, default="data/model/model_spo2.h5")
    args = parser.parse_args()
    print(args)

    df_static = pd.read_csv(config.get('processed', 'df_static_file'))
    df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))
    static_feat = pd.read_csv('../data/features/static-notxt.csv')

    timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, pos_rate = prepare_data(df_static, df_dynamic)
    # model, model_encoder = transition_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, args)
    # model.save(args.model_path)
    # model_encoder.save(args.encoder_path)
    # model_encoder = models.load_model(args.encoder_path)
    # model_predict = prediction_model(timeSeriesTr, labelsTr, timeSeriesTe, labelsTe, model_encoder, args)
    test_data_generator = PredictorDataGenerator(timeSeriesTe, static_feat, labelsTe,
                                        batch_size=len(labelsTe),
                                        window_size=args.steps,
                                        n_classes=2)
    spo2_data_generator = SpO2DataGenerator(timeSeriesTe, static_feat, labelsTe,
                                                   batch_size=len(labelsTe),
                                                   window_size=args.steps,
                                                   horizon=args.horizon,
                                                   shuffle=False)
    model = models.load_model(args.model_path)
    evaluate_generator(model, test_data_generator, spo2_data_generator)
    # evaluate(model_predict, test_data_generator, pos_rate)









