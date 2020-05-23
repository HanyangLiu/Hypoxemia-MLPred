
import keras
import numpy as np
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, timeSeries, static, labels, batch_size=1024, window_size=10, n_classes=2, shuffle=False):
        'Initialization'
        self.n_features = len(timeSeries.columns) + len(static.columns) - 3
        self.batch_size = batch_size
        self.timeSeries = timeSeries
        self.static = static
        self.labels = labels
        self.list_data_index = list(timeSeries['index'].values)
        self.n_classes = n_classes
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
        list_IDs_seed = list(range(max(list_IDs_temp[0] - self.sliding_window, 0), list_IDs_temp[0])) + list_IDs_temp
        # get df segment for this batch plus several rows before the first ID in this batch
        df_seed = self.timeSeries[self.timeSeries['index'].isin(list_IDs_seed)]
        for j in range(self.sliding_window):
            slice = pd.merge(df_seed.groupby(['pid']).shift(periods=j, fill_value=0).reset_index(level=0, drop=True),
                             self.static, how='left', on='pid')
            X[:, j, :] = slice.iloc[len(df_seed) - len(list_IDs_temp):, 2:]

        # y: (n_samples, 1)
        y = self.labels.loc[self.labels['index'].isin(list_IDs_temp), 'label'].values
        if len(y) != len(X):
            print("Error!!! - Shapes of X and y do not match!")

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


