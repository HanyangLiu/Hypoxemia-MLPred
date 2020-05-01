import numpy as np
from numpy import array
from numpy import hstack
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
import pickle
import sys


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.feature_window = 10

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


parser = argparse.ArgumentParser(description='hypoxemia prediction')
parser.add_argument('--hypoxemia_thresh', type=int, default=90)
parser.add_argument('--hypoxemia_window', type=int, default=10)
parser.add_argument('--prediction_window', type=int, default=5)
parser.add_argument('--filter_mode', type=str, default='exclude')
args = parser.parse_args()
print(args)


df_static = pd.read_csv(config.get('processed', 'df_static_file'))
df_dynamic = pd.read_csv(config.get('processed', 'df_dynamic_file'))


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

timeSeriesTr = df_dynamic.iloc[selected_idx_train, 2:]
labelsTr = dynamic_label.iloc[selected_idx_train, 'label']
timeSeriesTe = df_dynamic.iloc[selected_idx_test, 2:]
labelsTe = dynamic_label.iloc[selected_idx_test, 'label']



