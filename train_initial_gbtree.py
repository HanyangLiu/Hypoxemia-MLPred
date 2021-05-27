import pandas as pd
import numpy as np
from file_config.config import config
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation
from utils.utility_analysis import line_search_best_metric, au_prc
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from train_catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import shap
from train_initial_catboost import prepare_data


def train_gbtree(X_train, y_train):

    # Training
    print('Training model...')
    # shuffle X and y
    X_train, y_train = shuffle(X_train, y_train,
                               random_state=0
                               )
    if args.gb_tool == 'xgboost':
        model = XGBClassifier(objective='binary:logistic',
                              booster='gbtree',
                              learning_rate=0.05,
                              n_estimators=200,
                              max_depth=3,
                              min_child_weight=6,
                              verbosity=1,
                              )
        model.fit(X_train, y_train)
        params = model.get_params()
    else:
        model = CatBoostClassifier(verbose=0,
                                   cat_features=cat_features,
                                   random_state=args.rs_model,
                                   # scale_pos_weight=(1 - pos_rate) / pos_rate
                                   )
        model.fit(X_train, y_train)
        params = model.get_all_params()

    print('Parameters:', params)
    print('Done.')

    return model


def cv_train(X, y):
    X, y = shuffle(X, y, random_state=1)
    model = XGBClassifier(objective='binary:logistic', booster='gbtree') if args.gb_tool == 'xgboost' else CatBoostClassifier(verbose=0, cat_features=cat_features)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scoring = {
        'AU-ROC': metrics.make_scorer(metrics.roc_auc_score, needs_proba=True, greater_is_better=True),
        'AU-PRC': metrics.make_scorer(au_prc, needs_proba=True, greater_is_better=True),
    }
    results = cross_validate(model, X, y,
                             cv=kfold,
                             scoring=scoring,
                             return_train_score=True)
    print(results)
    print("AU-ROC: %.2f%% (%.2f%%)" % (results['test_AU-ROC'].mean() * 100, results['test_AU-ROC'].std() * 100))
    print("AU-PRC: %.2f%% (%.2f%%)" % (results['test_AU-PRC'].mean() * 100, results['test_AU-PRC'].std() * 100))


def param_tuning(X, y):

    # shuffle X and y
    X, y = shuffle(X, y,
                   random_state=0
                   )
    print('Searching best parameters...')
    if args.gb_tool == 'xgboost':
        model = XGBClassifier(objective='binary:logistic', booster='gbtree')
        param_dist = {"max_depth": [3],
                      "min_child_weight": [6],
                      "n_estimators": [100, 200, 1000],
                      "learning_rate": [0.05, 0.3], }
        """
        Best Parameters (Default): {'objective': 'binary:logistic', 'base_score': 0.5, 'booster': 'gbtree', 
        'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'gpu_id': -1, 
        'importance_type': 'gain', 'interaction_constraints': None, 'learning_rate': 0.05, 'max_delta_step': 0, 
        'max_depth': 3, 'min_child_weight': 6, 'missing': nan, 'monotone_constraints': None, 'n_estimators': 1000, 
        'n_jobs': 0, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 
        'subsample': 1, 'tree_method': None, 'validate_parameters': False, 'verbosity': 1}
        """
    else:
        model = CatBoostClassifier(verbose=0)
        param_dist = {'depth': [3, 6, 10],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'l2_leaf_reg': [1, 3, 6],
                      }
        """
        Best Parameters (Default): {'nan_mode': 'Min', 'eval_metric': 'Logloss', 'iterations': 1000, 
        'sampling_frequency': v    'PerTree', 'leaf_estimation_method': 'Newton', 'grow_policy': 'SymmetricTree', 
        'penalties_coefficient': 1, 'boosting_type': 'Plain', 'model_shrink_mode': 'Constant', 
        'feature_border_type': 'GreedyLogSum', 'bayesian_matrix_reg': 0.10000000149011612, 
        'l2_leaf_reg': 3, 'random_strength': 1, 'rsm': 1, 'boost_from_average': False, 
        'model_size_reg': 0.5, 'subsample': 0.800000011920929, 'use_best_model': False, 'class_names': [0, 1], 
        'random_seed': 0, 'depth': 6, 'border_count': 254, 'classes_count': 0, 'sparse_features_conflict_fraction': 0, 
        'leaf_estimation_backtracking': 'AnyImprovement', 'best_model_min_trees': 1, 'model_shrink_rate': 0, 
        'min_data_in_leaf': 1, 'loss_function': 'Logloss', 'learning_rate': 0.058687999844551086, 
        'score_function': 'Cosine', 'task_type': 'CPU', 'leaf_estimation_iterations': 10, 
        'bootstrap_type': 'MVS', 'max_leaves': 64}
        """
    grid_search = GridSearchCV(model,
                               param_grid=param_dist,
                               cv=5,
                               verbose=10,
                               scoring=metrics.make_scorer(au_prc, needs_proba=True, greater_is_better=True),
                               n_jobs=-1
                               )
    grid_search.fit(X, y)
    print('Best parameters:', grid_search.best_params_)
    model_best = grid_search.best_estimator_
    print('Done.')

    return model_best


def evaluate(model, X_test, y_test, pos_rate):

    # Testing
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    prec, rec, _ = metrics.precision_recall_curve(y_test, y_prob)
    (sensitivity, specificity, PPV, NPV, f1, acc), _ = line_search_best_metric(y_test, y_prob)
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
    result_table.to_pickle('data/result/initial_gbtree.pkl')

    # # plot ROC and PRC
    # plot_roc(fpr, tpr, 'data/result/roc_initial.png')
    # plot_prc(rec, prec, 'data/result/pr_initial.png')


def model_explain(model, data_to_predict):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, data_to_predict, plot_type="bar")
    vals = np.abs(shap_values).mean(0)
    feat_list = X_train.columns.to_list()
    for ind, feat in enumerate(feat_list):
        if feat in feat_dict:
            feat_list[ind] = feat_dict[feat]
        else:
            feat_list[ind] = feat

    importance = pd.DataFrame(list(zip(feat_list, vals)), columns=['Feature', 'Feature Importance'])
    for ind in importance.index:
        if ind > 9:
            importance.loc[ind, 'Feature'] = "Text-'" + importance.loc[ind, 'Feature'] + "'"
    importance.sort_values(by=['Feature Importance'], ascending=False, inplace=True)
    importance.set_index('Feature', inplace=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    barWidth = 0.5
    bars1 = list(importance[0:9]['Feature Importance'].values/importance['Feature Importance'].sum())
    r1 = np.arange(len(bars1))
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='PreOp Prediction')
    plt.xticks([r for r in range(len(bars1))], list(importance[0:20].index))
    fig.autofmt_xdate(rotation=45)
    plt.ylabel('Feature Impact', fontweight='bold')
    # plt.title('Feature Importance of Preoperative Model')
    plt.legend()
    plt.savefig('data/result/feat_importance_init.pdf')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypoxemia_thresh', type=int, default=90)
    parser.add_argument('--hypoxemia_window', type=int, default=10)
    parser.add_argument('--prediction_window', type=int, default=5)
    parser.add_argument('--static_feature_file', type=str, default='static-bow.csv')
    parser.add_argument('--random_state', type=int, default=2)
    parser.add_argument('--rs_model', type=int, default=0)
    parser.add_argument('--gb_tool', type=str, default='catboost')
    parser.add_argument('--if_tuning', type=str, default='False')
    parser.add_argument('--if_cv', type=str, default='False')
    args = parser.parse_args()
    print(args)

    X, y, pos_rate = prepare_data(df_static=pd.read_csv(config.get('processed', 'df_static_file')),
                                  df_dynamic=pd.read_csv(config.get('processed', 'df_dynamic_file')),
                                  static_feature=pd.read_csv('data/features/' + args.static_feature_file),
                                  args=args)
    if args.gb_tool == 'catboost':
        cat_features = np.array([0, 4, 5, 7, 8, 9])
        X.iloc[:, cat_features] = X.iloc[:, cat_features].astype('str')

    if args.if_cv == 'True':
        # cross validation
        cv_train(X, y)
    else:
        # normal validation
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=args.random_state,
                                                            stratify=y)
        model = param_tuning(X, y) if args.if_tuning == 'True' else train_gbtree(X_train, y_train)
        pickle.dump(model, open(config.get('processed', 'initial_model_file'), 'wb'))
        evaluate(model, X_test, y_test, pos_rate)
        model_explain(model, X_test)

        feat_dict = {'pid': 'Patient ID', 'Gender': 'Gender', 'AGE': 'Age', 'HEIGHT': 'Height',
                     'WEIGHT': 'Weight', 'SecondHandSmoke': 'Second-hand Smoke', 'BedName': 'Operating Room',
                     'TimeOfDay': 'Time of Day', 'AnesthesiaType': 'Anesthesia Type', 'ASA': 'ASA',
                     'if_Emergency': 'Emergency', 'Airway_1': 'First Airway', 'Airway_1_Time': 'First Airway Time',
                     'Airway_2': 'Second Airway', 'Airway_2_Time': 'Second Airway Time',
                     'AnesthesiaDuration': 'AnesthesiaDuration', 'EBL': 'Estimated Blood Loss',
                     'Urine_Output': 'Urine Output'}









