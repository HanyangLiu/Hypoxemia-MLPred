import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric, au_prc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
from train_catboost import CatBoostClassifier
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
    path_sta_label = 'data/result/static_label.pkl'
    path_dyn_label = 'data/result/dynamic_label.pkl'
    label_assign = LabelAssignment(hypoxemia_thresh=args.hypoxemia_thresh,
                                   hypoxemia_window=args.hypoxemia_window,
                                   prediction_window=args.prediction_window)
    # print('Assigning labels...')
    # static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
    # static_label.to_pickle(path_sta_label)
    # dynamic_label.to_pickle(path_dyn_label)

    static_label = pd.read_pickle(path_sta_label)
    dynamic_label = pd.read_pickle(path_dyn_label)
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
                                                 random_state=args.random_state,
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

    # adjust features used
    dynamic_feature = dynamic_feature.drop(columns=['AnesthesiaDuration', 'EBL', 'Urine_Output'])
    # column_names = list(dynamic_feature.columns)
    # drop_list = []
    # for name in column_names:
    #     if 'FiO2' in name or 'coreTemp' in name:
    #         drop_list.append(name)
    # dynamic_feature.drop(columns=drop_list)

    # split into training and test set
    X_train = dynamic_feature.iloc[selected_idx_train, 2:]
    X_test = dynamic_feature.iloc[selected_idx_test, 2:]
    y_train = dynamic_label.loc[selected_idx_train, 'label']
    y_test = dynamic_label.loc[selected_idx_test, 'label']

    # shuffle X and y
    X_train, y_train = shuffle(X_train, y_train,
                               random_state=0
                               )

    # positive number
    num_pos = np.sum(y_train) + np.sum(y_test)
    num_all = len(selected_idx_train) + len(selected_idx_test)
    pos_rate = num_pos/num_all
    print('Positive samples:', num_pos, '/', num_all)
    print('Ratio:', '%0.2f' % (num_pos/num_all*100), '%')

    return X_train, X_test, y_train, y_test, pos_rate


def param_tuning(X_train, y_train):

    # shuffle X and y
    X_train, y_train = shuffle(X_train, y_train,
                               random_state=0
                               )
    print('Searching best parameters...')

    model = CatBoostClassifier(verbose=0)
    param_dist = {
        'depth': [3, 6, 9],
        'learning_rate': [0.01, 0.02, 0.05],
        'l2_leaf_reg': [1, 3, 6],
        }

    grid_search = GridSearchCV(model,
                               param_grid=param_dist,
                               cv=3,
                               verbose=10,
                               scoring=metrics.make_scorer(au_prc, needs_proba=True, greater_is_better=True),
                               n_jobs=args.n_jobs
                               )
    grid_search.fit(X_train, y_train)
    print('Best parameters:', grid_search.best_params_)
    model_best = grid_search.best_estimator_
    print('Done.')

    return model_best


def evaluate(model, X_test, y_test, pos_rate, args):
    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]
    np.savetxt('data/result/y_prob', y_prob)
    np.savetxt('data/result/y_test', y_test)

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob, spec_thresh=0.95)
    alarm_rate = pos_rate * sensitivity / PPV

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
    print("Alarm rate:", alarm_rate)
    print('--------------------------------------------')

    result_table = pd.DataFrame(columns=['args', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])
    result_table = result_table.append({
        'args': args.__dict__,
        'fpr': fpr,
        'tpr': tpr,
        'roc': metrics.auc(fpr, tpr),
        'prec': prec,
        'rec': rec,
        'prc': metrics.auc(rec, prec),
        'y_test': y_test,
        'y_prob': y_prob,
        'pos_rate': pos_rate
    }, ignore_index=True)
    # save results
    result_table.to_pickle('data/result/realtime_gbtree.pkl')


    return metrics.auc(rec, prec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)

    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--l2', type=int, default=3)
    args = parser.parse_args()
    print(args)

    X_train, X_test, y_train, y_test, pos_rate = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                                              df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                                              dynamic_feature=pd.read_csv('data/features/' + args.feature_file),
                                                              args=args)

    model = param_tuning(X_train, y_train)
    pickle.dump(model, open(config.get('processed', 'realtime_model_file'), 'wb'))
    evaluate(model, X_test, y_test, pos_rate, args)




