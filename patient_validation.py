
import pickle
import pandas as pd

df_static = pd.read_csv('data/raw_data/static_updated.csv')
df_dynamic = pd.read_csv('data/data_frame/dynamic_dataframe.csv')
PatID2pid = pickle.load(open('data/result/dict_PatientID2pid.pkl', 'rb'))

PatID = 714595
df_dy = df_dynamic[df_dynamic['pid'] == PatID2pid[PatID]]
df_st = df_static[df_static['PatientID'] == PatID]


