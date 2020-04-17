import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import FeatureExtraction, DataImputation
from tsfresh.utilities.dataframe_functions import get_range_values_per_column
import argparse


def imputation(df_static, df_dynamic):
    # Missing value imputation
    print('Imputing missing values...')
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    if args.if_impute == 'True':
        df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
    print('Done!')

    return df_static, df_dynamic


def feature_extraction(df_static, df_dynamic, feat_dir):
    # Feature extraction
    print('Extracting static and real-time features...')
    extractor = FeatureExtraction(feature_window=5)
    df_static_features = extractor.gen_static_features(df_static, if_text=True)
    df_dynamic_features = extractor.gen_ewm_dynamic_features(df_static, df_dynamic, if_text=True)
    print('Done static and real-time feature extraction!')
    # save file
    print('Saving to files...')
    df_static_features.to_csv(static_feature_file, index=False)
    df_dynamic_features.to_csv(feat_dir, index=False)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument('--if_impute', type=str, default='True')
    args = parser.parse_args()
    # path
    df_static_file = config.get('processed', 'df_static_file')
    df_dynamic_file = config.get('processed', 'df_dynamic_file')
    static_feature_file = config.get('processed', 'static_feature_file')
    # load DataFrame real-time data
    df_static = pd.read_csv(df_static_file)
    df_dynamic = pd.read_csv(df_dynamic_file)

    print('Imputation:', args.if_impute)
    if args.if_impute == 'True':
        feat_dir = 'data/features/dynamic_feature_PCA.csv'
    else:
        feat_dir = 'data/features/dynamic_feature_not_imputed_PCA.csv'

    df_static, df_dynamic = imputation(df_static, df_dynamic)
    feature_extraction(df_static, df_dynamic, feat_dir)





