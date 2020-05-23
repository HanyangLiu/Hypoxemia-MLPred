import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import sys
import shap


def prepare_data(df_static, df_dynamic, dynamic_feature, args):

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

    del df_static, df_dynamic

    # select feature rows with pid in subgroup as data matrix
    print('Training/testing split:', len(pid_train), '/', len(pid_test))
    print('Split into training and test set...')
    to_keep = (dynamic_label['if_to_drop'] == 0).values
    is_in_train = dynamic_label[['pid']].isin(pid_train)['pid'].values
    is_in_test = dynamic_label[['pid']].isin(pid_test)['pid'].values
    selected_idx_train = list(np.where(to_keep & is_in_train)[0])
    selected_idx_test = list(np.where(to_keep & is_in_test)[0])

    # split into training and test set
    X_train = dynamic_feature.iloc[selected_idx_train, 2:].values
    X_test = dynamic_feature.iloc[selected_idx_test, 2:].values
    y_train = dynamic_label.loc[selected_idx_train, 'label'].values
    y_test = dynamic_label.loc[selected_idx_test, 'label'].values

    # shuffle X and y
    X_train, y_train = shuffle(X_train, y_train,
                               # random_state=0
                               )

    # positive number
    num_pos = np.sum(y_train) + np.sum(y_test)
    num_all = len(selected_idx_train) + len(selected_idx_test)
    print('Positive samples:', num_pos, '/', num_all)
    print('Ratio:', '%0.2f' % (num_pos/num_all*100), '%')

    return X_train, X_test, y_train, y_test


def train_gbtree(X_train, y_train):
    # Training
    print('Training model...')
    model = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          # silent=False,
                          # learning_rate=0.2,
                          # n_estimators=1000,
                          # max_depth=6,
                          # verbosity=2
                          )
    model.fit(X_train, y_train,
              # eval_metric=eval_metric,
              # eval_set=eval_set,
              verbose=True)
    print('Done.')

    return model


def evaluate(model, X_test, y_test):
    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]
    np.savetxt('data/result/y_prob', y_prob)
    np.savetxt('data/result/y_test', y_test)

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)

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
    print('--------------------------------------------')

    # plot ROC and PRC
    plot_roc(fpr, tpr, 'data/result/roc_initial.png')
    plot_prc(rec, prec, 'data/result/pr_initial.png')


def model_explain(model, data_to_predict):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_to_predict)
    shap.summary_plot(shap_values, data_to_predict, plot_type="bar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--feature_file', type=str, default='dynamic-ewm-notxt-imp.csv')
    args = parser.parse_args()
    print(args)

    X_train, X_test, y_train, y_test = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                                    df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                                    dynamic_feature=pd.read_csv('data/features/' + args.feature_file),
                                                    args=args)
    model = train_gbtree(X_train, y_train)
    pickle.dump(model, open(config.get('processed', 'realtime_model_file'), 'wb'))
    evaluate(model, X_test, y_test)


