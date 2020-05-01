# utilities for data preprocessing

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.impute import SimpleImputer, MissingIndicator
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from tqdm import tqdm, trange, tnrange
import tsfresh
import sys


class LabelAssignment:

    def __init__(self, hypoxemia_thresh=90, hypoxemia_window=10, prediction_window=5):
        self.hypoxemia_thresh = hypoxemia_thresh
        self.hypoxemia_window = hypoxemia_window
        self.prediction_window = prediction_window

    def get_positive_pids(self, static_labels):
        pos_pid_set = static_labels['pid'].values[static_labels['label'].values.astype(bool)]

        return pos_pid_set

    def assign_label(self, df_static, df_dynamic):

        win_hypo = self.hypoxemia_window
        win_pred = self.prediction_window
        SpO2_arr = df_dynamic['SpO2'].values
        if_SpO2_recorded = df_dynamic['if_SpO2'].values
        pid_arr = df_dynamic['pid'].values
        ts_arr = df_dynamic['ts'].values

        labels_static = np.zeros(len(df_static), dtype=bool)
        labels_dynamic = np.zeros(len(SpO2_arr), dtype=bool)

        if_low_SpO2 = np.zeros(len(SpO2_arr), dtype=bool)
        if_hypo_state = np.zeros(len(SpO2_arr), dtype=bool)
        if_to_drop = np.zeros(len(SpO2_arr), dtype=bool)

        # set if_hypo_state as 1 if currently under consecutive low SpO2
        for ind, SpO2 in enumerate(SpO2_arr):
            if ind + win_hypo >= len(SpO2_arr):
                continue
            if SpO2 <= self.hypoxemia_thresh and if_SpO2_recorded[ind] == 1:
                if_low_SpO2[ind] = True
                if np.sum([o2 <= self.hypoxemia_thresh for o2 in SpO2_arr[ind: ind + win_hypo]]) == win_hypo and \
                        pid_arr[ind] == pid_arr[ind + win_hypo]:
                    if_hypo_state[ind: ind + win_hypo] = True
                    labels_static[pid_arr[ind]] = True
            if ts_arr[ind] <= 10:
                if_to_drop[ind] = True
            if if_SpO2_recorded[ind] == 0:
                if_to_drop[ind - win_pred: ind] = True

        # label current timestep as 1 if any timestep within the future [win_pred] is under hypo_state
        for ind, SpO2 in enumerate(SpO2_arr):
            if ind + win_pred + 1 >= len(SpO2_arr):
                continue
            # True if any hypo_state within future [wind_pred] AND same pid [wind_pred] later
            if sum(if_hypo_state[ind + 1: ind + win_pred + 1]) > 0 and pid_arr[ind] == pid_arr[ind + win_pred]:
                labels_dynamic[ind] = True
            # to be discarded if label True AND currently under hypo_state
            if labels_dynamic[ind] and if_hypo_state[ind]:
                if_to_drop[ind] = True

        static_label = df_static[['pid']]
        static_label = static_label.assign(label=labels_static.astype(int))

        dynamic_label = df_dynamic[['index', 'pid', 'ts', 'SpO2']]
        dynamic_label = dynamic_label.assign(if_low_SpO2=if_low_SpO2.astype(int),
                                             label=labels_dynamic.astype(int),
                                             if_to_drop=if_to_drop.astype(int))

        return static_label, dynamic_label

    def assign_multi_label(self, df_static, df_dynamic):

        win_hypo = self.hypoxemia_window
        win_pred = self.prediction_window
        SpO2_arr = df_dynamic['SpO2'].values
        if_SpO2_recorded = df_dynamic['if_SpO2'].values
        pid_arr = df_dynamic['pid'].values
        ts_arr = df_dynamic['ts'].values

        labels_static = np.zeros(len(df_static), dtype=bool)
        labels_dynamic = np.zeros(len(SpO2_arr))

        if_low_SpO2 = np.zeros(len(SpO2_arr), dtype=bool)
        hypo_state = np.zeros(len(SpO2_arr))
        if_to_drop = np.zeros(len(SpO2_arr), dtype=bool)

        # set hypo_state as 1 if currently under consecutive low SpO2
        for ind, SpO2 in enumerate(SpO2_arr):
            if ind + win_hypo >= len(SpO2_arr):
                continue
            if SpO2 <= self.hypoxemia_thresh and if_SpO2_recorded[ind] == 1:
                if_low_SpO2[ind] = True
                if np.sum([o2 <= self.hypoxemia_thresh for o2 in SpO2_arr[ind: ind + win_hypo]]) == win_hypo and \
                        pid_arr[ind] == pid_arr[ind + win_hypo]:
                    hypo_state[ind: ind + win_hypo] = 2
                    labels_static[pid_arr[ind]] = True
            elif SpO2 >= 94:
                hypo_state[ind] = 0
            else:
                hypo_state[ind] = 1
            if ts_arr[ind] <= 10:
                if_to_drop[ind] = True
            if if_SpO2_recorded[ind] == 0:
                if_to_drop[ind - win_pred: ind] = True

        # label current timestep as 1 if any timestep within the future [win_pred] is under hypo_state
        if_hypo = hypo_state == 2
        for ind, SpO2 in enumerate(SpO2_arr):
            if ind + win_pred + 1 >= len(SpO2_arr):
                continue
            # True if any hypo_state within future [wind_pred] AND same pid [wind_pred] later
            if sum(if_hypo[ind + 1: ind + win_pred + 1]) > 0 and pid_arr[ind] == pid_arr[ind + win_pred]:
                labels_dynamic[ind] = 2
            else:
                labels_dynamic[ind] = hypo_state[ind + win_pred]
            # to be discarded if label True AND currently under hypo_state
            if labels_dynamic[ind] == 2 and hypo_state[ind] == 2:
                if_to_drop[ind] = True

        static_label = df_static[['pid']]
        static_label = static_label.assign(label=labels_static.astype(int))

        dynamic_label = df_dynamic[['index', 'pid', 'ts', 'SpO2']]
        dynamic_label = dynamic_label.assign(if_low_SpO2=if_low_SpO2.astype(int),
                                             label=labels_dynamic.astype(int),
                                             if_to_drop=if_to_drop.astype(int))

        return static_label, dynamic_label


class PatientFilter:

    def __init__(self, df_static, mode, include_icd, exclude_icd9, exclude_icd10):
        self.df_static = df_static  # DataFrame
        self.include_icd = include_icd  # list
        self.exclude_icd9 = exclude_icd9
        self.exclude_icd10 = exclude_icd10
        self.mode = mode

    def include_by_icd(self):

        included_pids = []

        for ind, pid in enumerate(self.df_static['pid'].values):
            icd9_str = self.df_static['ICD-9'].values[ind]
            icd10_str = self.df_static['ICD-10'].values[ind]

            if sum([code in str(icd9_str) for code in self.include_icd]) > 0 \
                    or sum([code in str(icd10_str) for code in self.include_icd]) > 0:
                included_pids.append(pid)

        return included_pids

    def exclude_by_icd(self):

        included_pids = []

        for ind, pid in enumerate(self.df_static['pid'].values):
            icd9_str = self.df_static['ICD-9'].values[ind]
            icd10_str = self.df_static['ICD-10'].values[ind]

            if sum([code in str(icd9_str) for code in self.exclude_icd9]) == 0 \
                    and sum([code in str(icd10_str) for code in self.exclude_icd10]) == 0:
                included_pids.append(pid)

        return included_pids

    def filter_by_icd(self):

        mode = self.mode

        if mode == 'include':
            included_pids = list(set(self.include_by_icd()) & set(self.exclude_by_icd()))
        elif mode == 'exclude':
            included_pids = self.exclude_by_icd()
        else:
            included_pids = None

        return included_pids


class DataImputation:

    def __init__(self):
        self.missing_gap_thresh = 10
        self.feat_name = ['invDiastolic', 'invMeanBP', 'invSystolic', 'HR', 'Diastolic', 'MeanBP', 'Systolic', 'SpO2',
                          'Pulse', 'ETCO2', 'Temp', 'coreTemp']
        self.inv_column = ['invDiastolic', 'invMeanBP', 'invSystolic']
        self.noninv_column = ['Diastolic', 'MeanBP', 'Systolic']

    def mean_impute(self, df, column_name):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = df[column_name].values
        imp.fit(X)
        imputed_array = imp.transform(X)
        df[column_name] = imputed_array

        return df

    def impute_static_dataframe(self, df_static):
        mean_column = ['AGE', 'HEIGHT', 'WEIGHT']
        df_static = self.mean_impute(df_static, mean_column)
        df_static = df_static.fillna(value=0)

        return df_static

    def impute_dynamic_dataframe(self, df_dynamic):

        # interpolate if gap is less than 10 timestep
        mask = df_dynamic[self.feat_name].copy()
        for column in self.feat_name:
            df = pd.DataFrame(df_dynamic[column])
            df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
            df['ones'] = 1
            mask[column] = (df.groupby('new')['ones'].transform('count')
                            < self.missing_gap_thresh) | df_dynamic[column].notnull()
        df_dynamic[self.feat_name] = df_dynamic[self.feat_name].interpolate().bfill()[mask]

        # add dummy variables
        indicator = MissingIndicator(missing_values=np.nan, features='all')
        X = df_dynamic[self.feat_name].values
        if_missing = indicator.fit_transform(X)
        if_measured = 1 - if_missing.astype(int)
        dummy_names = []
        for ind, feat in enumerate(self.feat_name):
            dummy_name = 'if_' + feat
            df_dynamic[dummy_name] = if_measured[:, ind]
            dummy_names.append(dummy_name)

        # impute missing invasive variables with 0 and add column "index"
        df_dynamic = df_dynamic.fillna(value=0)
        df_dynamic = df_dynamic.reindex(['index', 'pid', 'ts'] + self.feat_name + dummy_names, axis=1)

        return df_dynamic


class FeatureExtraction:

    def __init__(self, feature_window):
        feat_name = ['invDiastolic', 'invMeanBP', 'invSystolic', 'HR', 'Diastolic', 'MeanBP', 'Systolic', 'SpO2',
                     'Pulse', 'ETCO2', 'Temp', 'coreTemp']
        self.feat_use = feat_name
        self.sta_feature_columns = self.get_stat_feature_columns()
        self.ewm_feature_columns = self.get_ewm_feature_columns()
        self.feature_window = feature_window

    def get_ewm_feature_columns(self):

        feature_columns = []
        ewm_name = ['Last', 'min', 'max', 'EWMA-6s', 'EWMA-1m', 'EWMA-5m', 'EWMA-10m', 'EWMV-5m', 'EWMV-10']
        for ewm in ewm_name:
            for feat in self.feat_use:
                column = feat + '-' + ewm
                feature_columns.append(column)

        return feature_columns

    def get_stat_feature_columns(self):

        feature_columns = []
        stat_name = ['E', 'S', 'ro', 'skewness', 'kurtosis', 'trend', 'mean', 'min', 'max']
        for feat in self.feat_use:
            for stat in stat_name:
                column = feat + '-' + stat
                feature_columns.append(column)

        return feature_columns

    def get_sta_features(self, data):
        """
        Calculate the value of 9 kinds of selected statistical features
        :param data:
        :return:
        """

        def _cal_trend(data):
            time_list = np.arange(len(data))
            # create linear regression object
            regr = linear_model.LinearRegression()
            regr.fit(time_list.reshape(-1, 1), np.array(data).reshape(-1, 1))

            return regr.coef_[0][0]

        E = ts.abs_energy(data)
        S = ts.binned_entropy(data, max_bins=5)
        ro = ts.autocorrelation(data, lag=4)
        skewness = ts.skewness(data)
        kurtosis = ts.kurtosis(data)
        trend = _cal_trend(data)
        mean = ts.mean(data)
        min = ts.minimum(data)
        max = ts.maximum(data)

        return [E, S, ro, skewness, kurtosis, trend, mean, min, max]

    def gen_stat_dynamic_features(self, df_static, df_dynamic, sta_feat_file, feat_type):
        """
        Generate statistical features
        :param sta_feat_file:
        :param df_static:
        :param df_dynamic:
        :param feat_type:
        :return:
        """
        dynamic_columns = df_dynamic.columns
        num_feat = len(self.feat_use)
        pids = df_dynamic['pid'].unique()
        df_init = pd.DataFrame(data=None, columns=['index', 'pid', 'ts'] + self.sta_feature_columns)
        df_init.to_csv(sta_feat_file, index=False)

        for pid in tqdm(pids):
            df_current = df_dynamic[df_dynamic['pid'] == pid]
            feat_vecs = []
            for ind_df, index in enumerate(df_current['index'].values):
                if ind_df < self.feature_window - 1:
                    feat_vec = list(np.zeros(len(self.sta_feature_columns)))
                else:
                    feat_vec = []
                    for feature_name in self.feat_use:
                        time_series = df_current[feature_name].values[ind_df - self.feature_window + 1: ind_df + 1]
                        feat_vec += self.get_sta_features(time_series)
                feat_vecs.append(feat_vec)
            name_arr = df_current[['index', 'pid', 'ts']].values
            df = pd.DataFrame(data=np.concatenate([name_arr, np.array(feat_vecs)], axis=1),
                              columns=['index', 'pid', 'ts'] + self.sta_feature_columns)
            df.to_csv(sta_feat_file, mode='a', header=False, index=False)

        # normalization
        df = pd.read_csv(sta_feat_file)
        df = df.fillna(value=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        X = df[['ts'] + self.sta_feature_columns].values
        df[['ts'] + self.sta_feature_columns] = min_max_scaler.fit_transform(X)

        # for imputed data, add dummy columns
        if len(dynamic_columns) > num_feat + 3:
            dummy_columns = dynamic_columns[num_feat + 4:]
            df_dummy = df_dynamic[dummy_columns]
            df = pd.concat([df, df_dummy], axis=1, sort=False)

        # add static features to each row
        df = pd.merge(df, self.gen_static_features(df_static, feat_type=feat_type), on='pid')
        df.to_csv(sta_feat_file, index=False)

    def gen_ewm_dynamic_features(self, df_static, df_dynamic, feat_type):
        """
        Extracting exponentially weighted moving average/variance features.
        :param df_static:
        :param df_dynamic:
        :param feat_type:
        :return:
        """

        num_feat = len(self.feat_use)
        dynamic_columns = df_dynamic.columns
        ewm_columns = self.ewm_feature_columns

        df_seed = df_dynamic[self.feat_use].copy()
        df = df_dynamic[['index', 'pid', 'ts']].copy()

        df[ewm_columns[0:num_feat]] = df_seed
        df[ewm_columns[num_feat:num_feat * 2]] = df_seed.rolling(window=self.feature_window).min()
        df[ewm_columns[num_feat * 2:num_feat * 3]] = df_seed.rolling(window=self.feature_window).max()
        df[ewm_columns[num_feat * 3:num_feat * 4]] = df_seed.ewm(halflife=0.1).mean()
        df[ewm_columns[num_feat * 4:num_feat * 5]] = df_seed.ewm(halflife=1).mean()
        df[ewm_columns[num_feat * 5:num_feat * 6]] = df_seed.ewm(halflife=5).mean()
        df[ewm_columns[num_feat * 6:num_feat * 7]] = df_seed.ewm(halflife=10).mean()
        df[ewm_columns[num_feat * 7:num_feat * 8]] = df_seed.ewm(halflife=5).var()
        df[ewm_columns[num_feat * 8:num_feat * 9]] = df_seed.ewm(halflife=10).var()

        df = df.fillna(value=0)
        # normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X = df[['ts'] + self.ewm_feature_columns].values
        df[['ts'] + self.ewm_feature_columns] = min_max_scaler.fit_transform(X)

        # for imputed data, add dummy columns
        if len(dynamic_columns) > num_feat + 3:
            dummy_columns = dynamic_columns[num_feat + 4:]
            df_dummy = df_dynamic[dummy_columns]
            df = pd.concat([df, df_dummy], axis=1, sort=False)

        # add static features to each row
        df = pd.merge(df, self.gen_static_features(df_static, feat_type=feat_type), on='pid')

        return df

    def gen_lstm_features(self, df_static, df_dynamic, feat_type):
        """
        Extracting exponentially weighted moving average/variance features.
        :param df_static:
        :param df_dynamic:
        :param feat_type:
        :return:
        """

        num_feat = len(self.feat_use)
        dynamic_columns = df_dynamic.columns
        ewm_columns = self.ewm_feature_columns

        df_seed = df_dynamic[self.feat_use].copy()
        df = df_dynamic[['index', 'pid', 'ts']].copy()

        df[ewm_columns[0:num_feat]] = df_seed
        df[ewm_columns[num_feat:num_feat * 2]] = df_seed.rolling(window=self.feature_window).min()
        df[ewm_columns[num_feat * 2:num_feat * 3]] = df_seed.rolling(window=self.feature_window).max()
        df[ewm_columns[num_feat * 3:num_feat * 4]] = df_seed.ewm(halflife=0.1).mean()
        df[ewm_columns[num_feat * 4:num_feat * 5]] = df_seed.ewm(halflife=1).mean()
        df[ewm_columns[num_feat * 5:num_feat * 6]] = df_seed.ewm(halflife=5).mean()
        df[ewm_columns[num_feat * 6:num_feat * 7]] = df_seed.ewm(halflife=10).mean()
        df[ewm_columns[num_feat * 7:num_feat * 8]] = df_seed.ewm(halflife=5).var()
        df[ewm_columns[num_feat * 8:num_feat * 9]] = df_seed.ewm(halflife=10).var()

        df = df.fillna(value=0)
        # normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X = df[['ts'] + self.ewm_feature_columns].values
        df[['ts'] + self.ewm_feature_columns] = min_max_scaler.fit_transform(X)

        # for imputed data, add dummy columns
        if len(dynamic_columns) > num_feat + 3:
            dummy_columns = dynamic_columns[num_feat + 4:]
            df_dummy = df_dynamic[dummy_columns]
            df = pd.concat([df, df_dummy], axis=1, sort=False)

        # add static features to each row
        df = pd.merge(df, self.gen_static_features(df_static, feat_type=feat_type), on='pid')

        return df

    def gen_static_features(self, df_static, feat_type):
        """
        Generate static features.
        :param df_static:
        :param feat_type:
        :return:
        """

        static_feat_use = ['pid', 'AGE', 'TimeOfDay', 'AnesthesiaDuration', 'ASA', 'if_Eergency', 'HEIGHT', 'WEIGHT',
                           'Airway_1', 'Airway_1_Time', 'Airway_2', 'Airway_2_Time']
        df = df_static[static_feat_use].copy()

        # normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X = df[static_feat_use[1:]].values
        df[static_feat_use[1:]] = min_max_scaler.fit_transform(X)

        # bag-of-words
        if feat_type == 'bow':
            procedure_corpus = df_static['ScheduledProcedure'].astype('|S').values
            vectorizer = CountVectorizer()
            vecs = vectorizer.fit_transform(procedure_corpus).toarray()

            df_corpus = pd.DataFrame(vecs)
            df = pd.concat([df, df_corpus], axis=1, sort=False)

        # bag-of-words and reduced by PCA
        elif feat_type == 'rbow':
            procedure_corpus = df_static['ScheduledProcedure'].astype('|S').values
            vectorizer = CountVectorizer()
            vecs = vectorizer.fit_transform(procedure_corpus).toarray()

            # Use PCA to condense BOW into lower-dimentional vectors
            pca = PCA(0.95)
            pca.fit(vecs)
            vecs = pca.transform(vecs)

            df_corpus = pd.DataFrame(vecs)
            df = pd.concat([df, df_corpus], axis=1, sort=False)

        return df
