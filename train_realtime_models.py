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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from train_catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import argparse
import time
import itertools
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed


def prepare_data(df_static, df_dynamic, dynamic_feature, args):
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

    # exclude patients from OR14 and CCL "bedname"
    exclude_list = df_static[df_static['BedName'].isin([14., 23., 24])]['pid'].values.tolist()
    subgroup_pids = list(set(subgroup_pids) - set(exclude_list))

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
    dynamic_feature.drop(columns=['AnesthesiaDuration', 'EBL', 'Urine_Output'])

    # split into training and test set
    X_train = dynamic_feature.iloc[selected_idx_train, 2:].values
    X_test = dynamic_feature.iloc[selected_idx_test, 2:].values
    y_train = dynamic_label.loc[selected_idx_train, 'label'].values
    y_test = dynamic_label.loc[selected_idx_test, 'label'].values

    # shuffle X and y
    X_train, y_train = shuffle(X_train, y_train,random_state=0)

    # positive number
    num_pos = np.sum(y_train) + np.sum(y_test)
    num_all = len(selected_idx_train) + len(selected_idx_test)
    pos_rate = num_pos/num_all
    print('Positive samples:', num_pos, '/', num_all)
    print('Ratio:', '%0.2f' % (num_pos/num_all*100), '%')

    return X_train, X_test, y_train, y_test, pos_rate


def train_model(index_pair):
    (model_idx, rs) = index_pair

    classifiers = [
        CatBoostClassifier(random_state=rs, verbose=0, learning_rate=0.02, depth=6, l2_leaf_reg=3),
        LogisticRegression(random_state=rs, penalty='l2', n_jobs=-1),
        CalibratedClassifierCV(RandomForestClassifier(random_state=rs, n_jobs=-1)),
        CalibratedClassifierCV(LinearSVC(random_state=rs, max_iter=3000)),
        MLPClassifier(random_state=rs, max_iter=10000)
    ]
    model_names = [
        "CatBoost",
        "LogisticRegression",
        "RandomForest",
        "LinearSVC",
        "MLP",
    ]

    cls = classifiers[model_idx]
    result_table = pd.DataFrame(columns=['model', 'random_state', 'fpr', 'tpr', 'roc', 'prec', 'rec', 'prc', 'pos_rate'])
    print('Round', rs)
    print('Training:', cls.__class__.__name__)
    model = cls.fit(X_train, y_train)
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
        'model': model_names[model_idx],
        'random_state': rs,
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

    return result_table


def parallel_train(n_iters=10):
    n_models = 5
    index_pairs = itertools.product(range(n_models), range(n_iters))
    # parallelize
    n_cores = multiprocessing.cpu_count()
    result_list = Parallel(n_jobs=n_cores)(delayed(train_model)(index_pair) for index_pair in tqdm(index_pairs))
    results = pd.concat(result_list, axis=0)
    # save results
    save_name = 'data/result/model_comparison/realtime_models' + time.strftime("%Y%m%d-%H%M%S") +'.pkl'
    results.to_pickle(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--filter_mode', type=str, default='exclude')
    parser.add_argument('--feature_file', type=str, default='dynamic-ewm-notxt-imp.csv')
    parser.add_argument('--random_state', type=int, default=1)

    args = parser.parse_args()
    print(args)

    X_train, X_test, y_train, y_test, pos_rate = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                                              df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                                              dynamic_feature=pd.read_csv('data/features/' + args.feature_file),
                                                              args=args)
    parallel_train(n_iters=10)



