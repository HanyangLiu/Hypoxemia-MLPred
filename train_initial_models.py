import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_analysis import plot_roc, plot_prc, metric_eval, line_search_best_metric, au_prc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
from train_catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import sys
import shap


def prepare_data(df_static, df_dynamic, static_feature, args):

    # label assignment (according to imputed SpO2)
    imputer = DataImputation()
    df_static = imputer.impute_static_dataframe(df_static)
    df_dynamic = imputer.impute_dynamic_dataframe(df_dynamic)
    path_sta_label = 'data/label/static_label.pkl'
    path_dyn_label = 'data/label/dynamic_label.pkl'
    label_assign = LabelAssignment(hypoxemia_thresh=args.hypoxemia_thresh,
                                   hypoxemia_window=args.hypoxemia_window,
                                   prediction_window=args.prediction_window)

    # print('Assigning labels...')
    # static_label, dynamic_label = label_assign.assign_label(df_static, df_dynamic)
    # static_label.to_pickle(path_sta_label)
    # dynamic_label.to_pickle(path_dyn_label)

    static_label = pd.read_pickle(path_sta_label)
    positive_pids = label_assign.get_positive_pids(static_label)
    print('Done.')

    # get subgroup pids
    subgroup_pids = PatientFilter(df_static=df_static,
                                  mode='exclude',
                                  include_icd=None,
                                  exclude_icd9=['745', '746', '747'],
                                  exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()

    print('Positive Patient:', len(set(subgroup_pids) & set(positive_pids)), '/', len(subgroup_pids))
    print('Before trimming:', len(positive_pids), '/', len(df_static))
    print('Trimmed cases:', len(df_static) - len(subgroup_pids))
    pos_rate = len(set(subgroup_pids) & set(positive_pids)) / len(subgroup_pids)
    print('Positive rate:', pos_rate)

    # select features with pid in subgroup as data matrix, and split into training and test set
    selected_idx = subgroup_pids
    static_feature.drop(columns=['AnesthesiaDuration', 'Airway_1', 'Airway_1_Time',
                                 'Airway_2', 'Airway_2_Time', 'EBL', 'Urine_Output'])
    X = static_feature.iloc[selected_idx, 1:]
    y = static_label.loc[selected_idx, 'label']

    return X, y, pos_rate


def train_models(X_train, y_train, pos_rate):

    result_table = pd.DataFrame(columns=['random_state', 'model', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])

    for rs in range(10):
        classifiers = [
            LogisticRegression(random_state=rs, n_jobs=-1),
            CalibratedClassifierCV(RandomForestClassifier(random_state=rs, n_jobs=-1)),
            CalibratedClassifierCV(LinearSVC(random_state=rs, max_iter=3000)),
            # SGDClassifier(random_state=rs, n_jobs=-1),
            # DecisionTreeClassifier(random_state=rs),
            # GaussianNB(random_state=rs),
            # KNeighborsClassifier(n_neighbors=100, n_jobs=-1),
        ]
        models = [
            "LogisticRegression",
            "RandomForestClassifier",
            "LinearSVC",
            "SGDClassifier",
            "GaussianNB",
            "KNeighborsClassifier"
        ]
        for ind, cls in enumerate(classifiers):

            print('Round', rs)
            print('Training:', cls.__class__.__name__)
            model = cls.fit(X_train, y_train)

            if cls.__class__.__name__ == "SGDClassifier":
                y_prob = cls.decision_function(X_test)
            else:
                y_prob = model.predict_proba(X_test)[::, 1]

            # Evaluation
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
            prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)

            print('--------------------------------------------')
            print('Evaluation of test set:', cls.__class__.__name__)
            print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr),
                  "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))
            print('--------------------------------------------')

            result_table = result_table.append({
                'random_state': rs,
                'model': models[ind],
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

    save_name = 'data/result/initial_models_random.pkl'
    # save results
    result_table.to_pickle(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--static_feature_file', type=str, default='static-bow.csv')
    parser.add_argument('--random_state', type=int, default=1)

    args = parser.parse_args()
    print(args)

    X, y, pos_rate = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                  df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                  static_feature=pd.read_csv('data/features/' + args.static_feature_file),
                                  args=args)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=args.random_state,
                                                        stratify=y)
    train_models(X_train, y_train, pos_rate)



