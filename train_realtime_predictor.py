import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import sys


def prepare_data(df_static_file, df_dynamic_file, feature_file):
    # load data
    df_static = pd.read_csv(df_static_file)
    df_dynamic = pd.read_csv(df_dynamic_file)
    dynamic_feature = pd.read_csv(feature_file)

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
                                                 stratify=static_label.loc[subgroup_pids]['label'].values)
    pid_train = sorted(list(pid_train))
    pid_test = sorted(list(pid_test))

    print('Positive Patient:', len(set(subgroup_pids) & set(positive_pids)), '/', len(subgroup_pids))
    print('Before trimming:', len(positive_pids), '/', len(df_static))
    print('Trimmed cases:', len(df_static) - len(subgroup_pids))

    del df_static, df_dynamic


    # select features with pid in subgroup as data matrix, and split into training and test set
    selected_idx_train = []
    selected_idx_test = []
    num_train = len(pid_train)
    num_test = len(pid_test)
    print('Get training set index...')
    for i, pid in enumerate(pid_train):
        s = str(i + 1) + '/' + str(num_train)
        sys.stdout.write('\r' + s)
        df = dynamic_label[dynamic_label['pid'] == pid]
        selected_idx_train += list(df[df['if_to_drop'] == 0]['index'].values)
    print('\nGet test set index...')
    for i, pid in enumerate(pid_test):
        s = str(i + 1) + '/' + str(num_test)
        sys.stdout.write('\r' + s)
        df = dynamic_label[dynamic_label['pid'] == pid]
        selected_idx_test += list(df[df['if_to_drop'] == 0]['index'].values)
    print('\nDone.')

    print('Split into training and test set...')
    X_train = dynamic_feature.iloc[selected_idx_train, 2:].values
    X_test = dynamic_feature.iloc[selected_idx_test, 2:].values
    y_train = dynamic_label.loc[selected_idx_train, 'label'].values
    y_test = dynamic_label.loc[selected_idx_test, 'label'].values

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
                          silent=False,
                          # learning_rate=0.2,
                          # n_estimators=1000,
                          # max_depth=6,
                          # verbosity=2,
                          )
    eval_set = [(X_test, y_test)]
    eval_metric = ["aucpr"]
    model.fit(X_train, y_train, verbose=True)
    print('Done.')

    return model


def evaluate(model, X_test, y_test):
    # Testing
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, probs)
    prec, rec, _ = metrics.precision_recall_curve(y_test, probs)

    C = metrics.confusion_matrix(y_test, y_pred)
    tn = np.float(C[0][0])
    fn = np.float(C[1][0])
    tp = np.float(C[1][1])
    fp = np.float(C[0][1])

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = metrics.f1_score(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)

    print('Evaluation of test set:')
    print("AU-ROC:", "%0.4f" % metrics.auc(fpr, tpr), "AU-PRC:", "%0.4f" % metrics.auc(rec, prec))
    print("sensitivity:", "%0.4f" % sensitivity,
          "specificity:", "%0.4f" % specificity,
          "precision:", "%0.4f" % precision,
          "F1 score:", "%0.4f" % f1,
          "accuracy:", "%0.4f" % acc)

    plt.figure(0)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % metrics.auc(fpr, tpr))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('data/result/roc.png')

    plt.figure(1)
    plt.title('Precision Recall Curve')
    plt.plot(rec, prec, 'b', label='AUC = %0.2f' % metrics.auc(rec, prec))
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.savefig('data/result/pr.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--if_impute', type=str, default='True')
    parser.add_argument('--filter_mode', type=str, default='exclude')
    args = parser.parse_args()
    print(args)

    if args.if_impute == 'True':
        feat_dir = 'data/features/dynamic_feature.csv'
    else:
        feat_dir = 'data/features/dynamic_feature_not_imputed.csv'

    X_train, X_test, y_train, y_test = prepare_data(df_static_file=config.get('processed', 'df_static_file'),
                                                    df_dynamic_file=config.get('processed', 'df_dynamic_file'),
                                                    feature_file=feat_dir)
    model = train_gbtree(X_train, y_train)
    pickle.dump(model, open(config.get('processed', 'realtime_model_file'), 'wb'))
    evaluate(model, X_test, y_test)


