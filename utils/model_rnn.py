from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, Convolution1D
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf


def lstm_1(window_size, n_features, n_classes):

    model = Sequential()
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, input_shape=(window_size, n_features)))
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


def lstm_2(window_size, n_features, n_classes):

    model = Sequential()
    model.add(
        LSTM(128, dropout=0.3, recurrent_dropout=0.3, input_shape=(window_size, n_features), return_sequences=True))
    model.add(LSTM(128, activation='relu', unroll=True))
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


def lstm_3(window_size, n_features, n_classes):

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, n_features)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution1D(n_classes, kernel_size=1, activation="softmax", padding="same"))
    model.build(input_shape=(window_size, n_features))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    return model