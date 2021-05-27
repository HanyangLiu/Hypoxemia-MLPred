import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.utility_preprocess import PatientFilter, LabelAssignment, DataImputation


def analyze_cat(feat_name, dataframe):
    print('------{}-----'.format(feat_name))
    ls = dataframe[feat_name].unique()
    for item in ls:
        df = dataframe[dataframe[feat_name] == item]
        if item == 'nan':
            continue
        print("{}: {:.0f} ({:.0f}%)".format(item, len(df), len(df) / len(dataframe) * 100))
    print("NaN: {:.0f} ({:.0f}%)".format(len(df), len(df) / len(dataframe) * 100))
    print('---------------')


def analyze_cohort(df_static, df_raw):
    print('Cohort----------------\n-----------------')
    # correct negative ages
    a = df_static[df_static['AGE'] < 0]
    a['AGE'] += 100
    df_static[df_static['AGE'] <= 0] = a

    des_raw = df_raw.describe(include='all').transpose()
    des_sta = df_static.describe(include='all').transpose()

    # age
    print("Age: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['AGE', '50%'],
                                                des_sta.loc['AGE', '25%'],
                                                des_sta.loc['AGE', '75%']))

    # Sex
    print("Male sex: {} ({:.0f})".format(des_raw.loc['Gender', 'freq'],
                                         des_raw.loc['Gender', 'freq'] / des_raw.loc['PatientID', 'count'] * 100))

    # Height
    print("Height: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['HEIGHT', '50%'],
                                                   des_sta.loc['HEIGHT', '25%'],
                                                   des_sta.loc['HEIGHT', '75%']))

    # Weight
    print("Weight: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['WEIGHT', '50%'] / 1000,
                                                   des_sta.loc['WEIGHT', '25%'] / 1000,
                                                   des_sta.loc['WEIGHT', '75%'] / 1000))

    # BMI
    print("BMI: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['BMI', '50%'],
                                                des_sta.loc['BMI', '25%'],
                                                des_sta.loc['BMI', '75%']))

    # ASA physical status
    analyze_cat('ASA', df_static)

    # ASA emergency
    analyze_cat('if_Emergency', df_static)

    # Second Hand Smoke
    analyze_cat('SecondHandSmoke', df_static)

    # Anesthesia Type
    analyze_cat('AnesthesiaType', df_raw)

    # First Airway Event
    analyze_cat('Airway_Event1', df_raw)

    # Second Airway Event
    analyze_cat('Airway_Event2', df_raw)

    # EBL
    print("EBL: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['EBL', '50%'],
                                                des_sta.loc['EBL', '25%'],
                                                des_sta.loc['EBL', '75%']))

    # Urine
    print("Urine: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['Urine_Output', '50%'],
                                                  des_sta.loc['Urine_Output', '25%'],
                                                  des_sta.loc['Urine_Output', '75%']))

    # Anesthesia Duration
    print("Anesthesia Duration: {:.0f} ({:.0f}, {:.0f})".format(des_sta.loc['AnesthesiaDuration', '50%'],
                                                                des_sta.loc['AnesthesiaDuration', '25%'],
                                                                des_sta.loc['AnesthesiaDuration', '75%']))


raw = pd.read_csv('../data/raw_data/static_updated.csv')
static = pd.read_csv('../data/data_frame/static_dataframe.csv')
dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')
static['BMI'] = (static['WEIGHT'] / 1000) / (static['HEIGHT'] / 100) / (static['HEIGHT'] / 100)

print('Assigning labels...')
imputer = DataImputation()
df_static = imputer.impute_static_dataframe(static)
df_dynamic = imputer.impute_dynamic_dataframe(dynamic)
label_assign = LabelAssignment(hypoxemia_thresh=90,
                               hypoxemia_window=10,
                               prediction_window=5)
static_label = pd.read_pickle('../data/label/static_label.pkl')
dynamic_label = pd.read_pickle('../data/label/dynamic_label.pkl')
positive_pids = label_assign.get_positive_pids(static_label)
print('Done.')

# get subgroup pids
subgroup_pids = PatientFilter(df_static=df_static,
                              mode='exclude',
                              include_icd=None,
                              exclude_icd9=['745', '746', '747'],
                              exclude_icd10=['Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26']).filter_by_icd()
static = pd.read_csv('../data/data_frame/static_dataframe.csv')
static['BMI'] = (static['WEIGHT'] / 1000) / (static['HEIGHT'] / 100) / (static['HEIGHT'] / 100)
dynamic = pd.read_csv('../data/data_frame/dynamic_dataframe.csv')
X = static.iloc[subgroup_pids, :]
X_raw = raw.iloc[subgroup_pids, :]
y = static_label.loc[subgroup_pids, 'label']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                test_size=0.5,
                                                random_state=1,
                                                stratify=y_test)
pids_train = y_train.index
pids_val = y_val.index
pids_test = y_test.index
print('Overall:', len(X), 'Train:', len(pids_train), 'Test:', len(pids_test))

analyze_cohort(static.iloc[pids_train, :], raw.iloc[pids_train, :])
analyze_cohort(static.iloc[pids_val, :], raw.iloc[pids_val, :])
analyze_cohort(static.iloc[pids_test, :], raw.iloc[pids_test, :])
analyze_cohort(X, X_raw)

print('Overall positive:', len(X[y == 1]), 'Train:', np.sum([y_train == 1]), 'Validation:', np.sum([y_val == 1]), 'Test:', np.sum([y_test == 1]))

len(dynamic_label[dynamic_label['pid'].isin(pids_train)])
len(dynamic_label[dynamic_label['pid'].isin(pids_test)])

neg_pids = list(set(subgroup_pids) - set(positive_pids))
analyze_cohort(static.iloc[neg_pids, :], raw.iloc[neg_pids, :])

pos_pids = list(set(subgroup_pids) - set(neg_pids))
analyze_cohort(static.iloc[pos_pids, :], raw.iloc[pos_pids, :])

# select feature rows with pid in subgroup as data matrix
print('Training/testing split:', len(pids_train), '/', len(pids_test))
print('Split into training and test set...')
to_keep = (dynamic_label['if_to_drop'] == 0).values
is_in_train = dynamic_label[['pid']].isin(pids_train)['pid'].values
is_in_val = dynamic_label[['pid']].isin(pids_val)['pid'].values
is_in_test = dynamic_label[['pid']].isin(pids_test)['pid'].values
selected_idx_train = list(np.where(to_keep & is_in_train)[0])
selected_idx_val = list(np.where(to_keep & is_in_val)[0])
selected_idx_test = list(np.where(to_keep & is_in_test)[0])

# split into training and test set
y_train = dynamic_label.loc[selected_idx_train, 'label'].values
y_val = dynamic_label.loc[selected_idx_val, 'label'].values
y_test = dynamic_label.loc[selected_idx_test, 'label'].values

print(np.sum(y_train), np.sum(y_val), np.sum(y_test))
