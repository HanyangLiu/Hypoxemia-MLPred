import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import add_label
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from datetime import datetime
import matplotlib.pyplot as plt
import pyEX as p

####
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
# download_robot_execution_failures()
# timeseries, y = load_robot_execution_failures()
#
# extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
# impute(extracted_features)
#
# features_filtered = select_features(extracted_features, y)
#
# features_filtered_direct = extract_relevant_features(timeseries, y,
#                                                      column_id='id', column_sort='time')
#####


ticker = 'AMD'
timeframe = '1y'
df = p.chartDF(ticker, timeframe)
df = df[['close']]
df.reset_index(level=0, inplace=True)
df.columns=['ds','y']
plt.plot(df.ds, df.y)
plt.show()



# is_meanbp_nan = np.isnan(df_dynamic['MeanBP'].values)
# aa = np.zeros(len(df_dynamic))
# for i in range(self.partition_thresh):
#     aa += np.concatenate([np.zeros(i), is_meanbp_nan[0:len(df_dynamic) - i]])
# is_dense_measure = 1 - (aa >= self.partition_thresh - 1)



# pids = df_dynamic['pid'].unique()
# partition_thresh = self.partition_thresh
# noninv_indicators = []
# for pid in pids:
#     print(pid)
#     df = df_dynamic[df_dynamic['pid'] == pid]
#     mean_bp = df['MeanBP'].values
#     indicators = []
#     for ind, ts in enumerate(df['ts'].values):
#         if ind < partition_thresh - 1:
#             indicators.append(1)
#         elif np.sum([np.isnan(a)
#                      for a in mean_bp[ind - partition_thresh + 1: ind + 1]]) >= partition_thresh - 1:
#             indicators.append(0)
#             # df.loc[(df['ts'] == ts), noninv_column] = np.array([[0, 0, 0], ])
#         else:
#             indicators.append(1)
#
#     df_dynamic[df_dynamic['pid'] == pid] = df
#     noninv_indicators += indicators


# is_dense_measure = 1 - if_invasive[:, 1]
# df_dynamic = df_dynamic.assign(is_noninv_dense=is_dense_measure)
# is_not_null = - df_dynamic['MeanBP'].isnull()
#
# # impute missing noninvasive variables
# df_dynamic[noninv_column] = df_dynamic[noninv_column].interpolate(method='linear', limit_direction='forward',
#                                                                   limit_area='inside', axis=0)
# # df_dynamic[noninv_column] = df_dynamic[noninv_column].interpolate(method='linear', limit_direction='backward',
# #                                                                   axis=0)
# df_dynamic.loc[is_not_null, ['is_noninv_dense']] = 1
# df_dynamic.loc[df_dynamic['is_noninv_dense'] == 0, noninv_column] = 0