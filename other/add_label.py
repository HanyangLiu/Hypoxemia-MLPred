import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import add_label


# path
processed_static_file = config.get('processed', 'processed_static_file')
processed_realtime_file = config.get('processed', 'processed_realtime_file')
labeled_static_file = config.get('processed', 'labeled_static_file')
labeled_realtime_file = config.get('processed', 'labeled_realtime_file')

# parameter
hypoxemia_thresh = 90
hypoxemia_window = 10
prediction_window = 5

# load DataFrame real-time data
df_static = pd.read_csv(processed_static_file)
df_dynamic = pd.read_csv(processed_realtime_file)

# label each timestep by whether hypoxemia occur within the future window; label those pid if hypoxemia occurred
static_labeled, dynamic_labeled = add_label(df_static, df_dynamic,
                                            hypoxemia_thresh,
                                            hypoxemia_window,
                                            prediction_window)

# save DataFrame to csv
static_labeled.to_csv(labeled_static_file, index=False)
dynamic_labeled.to_csv(labeled_realtime_file, index=False)




