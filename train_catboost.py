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
from catboost import CatBoostClassifier
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
    path_sta_label = 'data/result/static_label_' + str(args.hypoxemia_window) + '.pkl'
    path_dyn_label = 'data/result/dynamic_label_' + str(args.hypoxemia_window) + '.pkl'
    label_assign = LabelAssignment(hypoxemia_thresh=args.hypoxemia_thresh,
                                   hypoxemia_window=args.hypoxemia_window,
                                   prediction_window=args.prediction_window)

    if os.path.exists(path_sta_label) and os.path.exists(path_dyn_label):
        static_label = pd.read_pickle(path_sta_label)
        dynamic_label = pd.read_pickle(path_dyn_label)
    else:
        print('Assigning labels...')
        static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
        static_label.to_pickle(path_sta_label)
        dynamic_label.to_pickle(path_dyn_label)
        print('Done.')

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
                                                 test_size=0.2,
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
    is_in_train = dynamic_label[['pid']].isin(pid_train)['pid'].values
    is_in_test = dynamic_label[['pid']].isin(pid_test)['pid'].values
    selected_idx_train = list(np.where(is_in_train)[0])
    selected_idx_test = list(np.where(is_in_test)[0])

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
    labelsTr = dynamic_label.iloc[selected_idx_train][['index', 'label', 'if_to_drop']]
    labelsTe = dynamic_label.iloc[selected_idx_test][['index', 'label', 'if_to_drop']]

    # shuffle X and y
    X_train, labelsTr = shuffle(X_train, labelsTr,
                               random_state=0
                               )

    # positive number
    num_pos = np.sum(labelsTr['label']) + np.sum(labelsTe['label'])
    num_all = len(selected_idx_train) + len(selected_idx_test)
    pos_rate = num_pos/num_all
    print('Positive samples:', num_pos, '/', num_all)
    print('Ratio:', '%0.2f' % (num_pos/num_all*100), '%')

    return X_train, X_test, labelsTr, labelsTe, pos_rate


def train_gbtree(X_train, labelsTr, pos_rate, args, labelsTe):

    y_train = labelsTr['label']

    X_train, labelsTr = shuffle(X_train, y_train,
                                random_state=0
                                )

    result_table = pd.DataFrame(columns=['random_state', 'model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])

    for rs in range(1):
        classifiers = [
            CatBoostClassifier(verbose=0,
                               # scale_pos_weight=(1 - pos_rate) / pos_rate,
                               learning_rate=args.lr,
                               depth=args.depth,
                               l2_leaf_reg=args.l2,
                               random_state=rs
                               )
        ]
        for cls in classifiers:

            print('Round', rs)
            print('Training:', cls.__class__.__name__)
            model = cls.fit(X_train, labelsTr)
            y_prob = model.predict_proba(X_test)[::, 1]
            y_test = labelsTe['label']
            labelsTe = labelsTe.assign(y_prob=y_prob)
            y_prob_w_hypo = labelsTe[labelsTe.if_to_drop == 0]['y_prob'].values
            y_test_w_hypo = labelsTe[labelsTe.if_to_drop == 0]['label'].values

            # Evaluation 1
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
            prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)

            print('--------------------------------------------')
            print('Evaluation of full test set:')
            print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
                  "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))

            # Evaluation 2
            fpr, tpr, _ = metrics.roc_curve(y_test_w_hypo, y_prob_w_hypo)
            prec, rec, _ = metrics.precision_recall_curve(y_test_w_hypo, y_prob_w_hypo)

            print('Evaluation of test set without hypoxemia samples:')
            print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
                  "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))
            print('--------------------------------------------')

            (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test_w_hypo, y_test_w_hypo, spec_thresh=0.95)
            alarm_rate = pos_rate * sensitivity / PPV

            print("sensitivity:", "%0.4f" % sensitivity,
                  "specificity:", "%0.4f" % specificity,
                  "PPV:", "%0.4f" % PPV,
                  "NPV:", "%0.4f" % NPV,
                  "F1 score:", "%0.4f" % f1,
                  "accuracy:", "%0.4f" % acc)
            print("Alarm rate:", alarm_rate)
            print('--------------------------------------------')

            # result_table = result_table.append({
            #     'random_state': rs,
            #     'model': cls.__class__.__name__,
            #     'fpr': fpr,
            #     'tpr': tpr,
            #     'roc': metrics.auc(fpr, tpr),
            #     'prec': prec,
            #     'rec': rec,
            #     'prc': metrics.auc(rec, prec),
            #     'y_test': labelsTe,
            #     'y_prob': y_prob,
            #     'pos_rate': pos_rate
            # }, ignore_index=True)

    save_name = 'data/result/model_comparison/realtime_gbtree_random.pkl'
    # save results
    # result_table.to_pickle(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--feature_file', type=str, default='dynamic-ewm-notxt-nonimp.csv')
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=-1)

    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--l2', type=int, default=3)
    args = parser.parse_args()
    print(args)

    X_train, X_test, labelsTr, labelsTe, pos_rate = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                                                 df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                                                 dynamic_feature=pd.read_csv('data/features/' + args.feature_file),
                                                                 args=args)

    train_gbtree(X_train, labelsTr, pos_rate, args, labelsTe)




