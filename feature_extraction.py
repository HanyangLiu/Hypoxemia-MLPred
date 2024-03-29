import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import FeatureExtraction, DataImputation
import argparse


def feature_extraction(df_static, df_dynamic, type):
    # Feature extraction
    print('Extracting features...')
    extractor = FeatureExtraction(feature_window=5)
    if type == 'static':
        df_static_features = extractor.gen_static_features(df_static, feat_type=args.static_txt)
        print('Done static feature extraction!')
        print('Saving to files:', static_feature_file)
        df_static_features.to_csv(static_feature_file, index=False)
        print('Done!')
    if type == 'dynamic-ewm':
        df_dynamic_features = extractor.gen_ewm_dynamic_features(df_static, df_dynamic, feat_type=args.dynamic_txt)
        print('Done real-time EWM feature extraction!')
        print('Saving to files:', ewm_feat_file)
        df_dynamic_features.to_csv(ewm_feat_file, index=False)
        print('Done!')
    if type == 'dynamic-sta':
        extractor.gen_stat_dynamic_features(df_static, df_dynamic, sta_feat_file, feat_type=args.dynamic_txt)
        print('Done real-time statistical feature extraction!')
    if type == 'dynamic-lstm':
        df = extractor.gen_lstm_features(df_static, df_dynamic, feat_type=args.dynamic_txt, sliding_window=10)
        print('Done real-time statistical feature extraction!')
        print('Saving to files:', lstm_feat_file)
        df.to_csv(lstm_feat_file, index=False)
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument('--type', type=str, default='static')  # 'static' OR 'dynamic-ewm' OR 'dynamic-sta'
    parser.add_argument('--if_impute', type=str, default='True')  # 'True' OR 'False'
    parser.add_argument('--static_txt', type=str, default='rbow')  # 'bow' OR 'rbow'
    parser.add_argument('--dynamic_txt', type=str, default='notxt')  # 'notxt' OR 'rbow'
    args = parser.parse_args()
    print(args)

    # path
    df_static_file = config.get('processed', 'df_static_file')
    df_dynamic_file = config.get('processed', 'df_dynamic_file')

    # save name
    token_impute = 'imp' if args.if_impute == 'True' else 'nonimp'
    static_feature_file = 'data/features/static-' + args.static_txt + '.csv'
    ewm_feat_file = 'data/features/dynamic-ewm-' + args.dynamic_txt + '-' + token_impute + '.csv'
    sta_feat_file = 'data/features/dynamic-sta-' + args.dynamic_txt + '-' + token_impute + '.csv'
    lstm_feat_file = 'data/features/dynamic-lstm-' + args.dynamic_txt + '-' + token_impute + '.csv'

    # load DataFrame real-time data
    df_static = pd.read_csv(df_static_file)
    df_dynamic = pd.read_csv(df_dynamic_file)

    # feature extraction
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic) if args.if_impute == 'True' else df_dynamic
    feature_extraction(df_static, df_dynamic, type=args.type)






