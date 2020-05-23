from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf


def lstm_1(window_size, n_features):

    model = Sequential()
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, input_shape=(window_size, n_features)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=[tf.keras.metrics.AUC(),
                 tf.keras.metrics.PrecisionAtRecall(0.5)
                 ]
    )

    return model


def lstm_2(window_size, n_features):

    model = Sequential()
    model.add(
        LSTM(128, dropout=0.3, recurrent_dropout=0.3, input_shape=(window_size, n_features), return_sequences=True))
    model.add(LSTM(128, activation='relu', unroll=True))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=[tf.keras.metrics.AUC(),
                 tf.keras.metrics.PrecisionAtRecall(0.5)
                 ]
    )

    return model