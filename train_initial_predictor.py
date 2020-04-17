import os
import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from imblearn.metrics import sensitivity_specificity_support
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import argparse
import pickle


def prepare_data(df_static_file, df_dynamic_file, feature_file, args):

    # load data
    df_static = pd.read_csv(df_static_file)
    df_dynamic = pd.read_csv(df_dynamic_file)
    static_feature = pd.read_csv(feature_file)

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
                                  mode='exclude',
                                  include_icd=None,
                                  exclude_icd9=['745', '746', '747'],
                                  exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()

    print('Positive Patient:', len(set(subgroup_pids) & set(positive_pids)), '/', len(subgroup_pids))
    print('Before trimming:', len(positive_pids), '/', len(df_static))
    print('Trimmed cases:', len(df_static) - len(subgroup_pids))

    # select features with pid in subgroup as data matrix, and split into training and test set
    selected_idx = subgroup_pids
    X = static_feature.iloc[selected_idx, 1:].values
    y = static_label.loc[selected_idx, 'label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def train_gbtree(X_train, y_train, X_test, y_test):

    # Training
    print('Training model...')
    model = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          silent=False,
                          # learning_rate=0.1,
                          # n_estimators=2000,
                          # max_depth=4,
                          # verbosity=2
                          )
    eval_set = [(X_test, y_test)]
    eval_metric = ["aucpr"]
    model.fit(X_train, y_train,
              # eval_metric=eval_metric,
              # eval_set=eval_set,
              verbose=True)
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
    plt.savefig('data/result/roc_initial.png')

    plt.figure(1)
    plt.title('Precision Recall Curve')
    plt.plot(rec, prec, 'b', label='AUC = %0.2f' % metrics.auc(rec, prec))
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.savefig('data/result/pr_initial.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='hypoxemia prediction')
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data(df_static_file=config.get('processed', 'df_static_file'),
                                                    df_dynamic_file=config.get('processed', 'df_dynamic_file'),
                                                    feature_file=config.get('processed', 'static_feature_file'),
                                                    args=args)
    model = train_gbtree(X_train, y_train, X_test, y_test)
    pickle.dump(model, open(config.get('processed', 'initial_model_file'), 'wb'))
    evaluate(model, X_test, y_test)


