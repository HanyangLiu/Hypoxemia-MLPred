
from datetime import datetime
import pandas as pd
import numpy as np
import os
import glob


class ParseICD:

    def __init__(self, id_converter, icd_file):
        self.idConvert_dict = id_converter
        self.icd_file = icd_file
        self.column_name = ['pid', 'ICD-9', 'present_flag_9', 'ICD-10', 'present_flag_10']

    def parse_icd(self):
        data_comb = {}
        with open(self.icd_file, 'r') as f:
            lines = f.readlines()
            for line_ind, line in enumerate(lines[1:]):
                feature_list = line.split(',')
                pid_ori = feature_list[0]
                if int(pid_ori) not in self.idConvert_dict:
                    continue
                pid = str(self.idConvert_dict[int(pid_ori)])
                icd_version = feature_list[2]
                icd = feature_list[3]
                present_flag = feature_list[-1]

                # fill data into dictionary
                if pid not in data_comb.keys():
                    data_comb[pid] = {}

                if str(icd_version) == '9-CM      ':
                    if 'ICD-9' not in data_comb[pid].keys():
                        data_comb[pid]['ICD-9'] = []
                    if 'present_flag_9' not in data_comb[pid].keys():
                        data_comb[pid]['present_flag_9'] = []
                    data_comb[pid]['ICD-9'].append(icd)
                    data_comb[pid]['present_flag_9'].append(present_flag[0:-1])

                if str(icd_version) == '10-CM     ':
                    if 'ICD-10' not in data_comb[pid].keys():
                        data_comb[pid]['ICD-10'] = []
                    if 'present_flag_10' not in data_comb[pid].keys():
                        data_comb[pid]['present_flag_10'] = []
                    data_comb[pid]['ICD-10'].append(icd)
                    data_comb[pid]['present_flag_10'].append(present_flag[0:-1])

        # Convert ICD dictionary into DataFrame
        rows = []
        for pid in data_comb:
            patient_data = data_comb[pid]
            patient_data['pid'] = int(pid)
            rows.append(patient_data)
        df = pd.DataFrame(rows)
        df = df.reindex(self.column_name, axis=1)
        df.sort_values(by=['pid'], axis=0, inplace=True)

        return df


class ParseStaticData:

    def __init__(self, demographic_file):
        self.demographic_file = demographic_file
        self.df_demo = self.parse_demographic()
        self.start_time_dict = self.extract_start_time()
        self.stop_time_dict = self.extract_stop_time()
        self.id_converter = self.extract_id_converter()  # patientID to pid

    def parse_demographic(self):
        df = pd.read_csv(self.demographic_file)
        df = df.sort_values(by=['PatientID'])

        return df

    def gen_static_dataframe(self):

        df_old = self.df_demo
        column_kept = ['HEIGHT', 'WEIGHT', 'SecondHandSmoke', 'ScheduledProcedure', 'EBL', 'Urine_Output']
        df = df_old.filter(column_kept, axis=1)

        # add converted pid column
        pids = np.array(list(self.id_converter.values()))
        df = df.assign(pid=pids)

        # add converted time
        durations, ages, airway1, airway2, time_of_day = self.time_abs_to_relative()

        # convert airway event to number code
        airway1_type = df_old['Airway_Event1'].unique()
        airway2_type = df_old['Airway_Event2'].unique()
        airway_type = list(set(airway1_type) | set(airway2_type))
        airway_mapping = dict(zip(airway_type, list(range(len(airway_type)))))
        airway1_code = [airway_mapping[ele] for ele in df_old['Airway_Event1'].values]
        airway2_code = [airway_mapping[ele] for ele in df_old['Airway_Event2'].values]

        df_old[['ASA']] = df_old[['ASA']].fillna(value=0)
        ASA = [str(el)[0] for _, el in enumerate(df_old['ASA'].values)]
        if_Emergency = ['E' in str(el) for _, el in enumerate(df_old['ASA'].values)]

        df = df.assign(AnesthesiaDuration=np.array(durations),
                       AGE=np.array(ages),
                       Gender=np.array(self.digitalize_val(df_old, 'Gender')),
                       AnesthesiaType=np.array(self.digitalize_val(df_old, 'AnesthesiaType')),
                       BedName=np.array(self.digitalize_val(df_old, 'BedName')),
                       Airway_1_Time=np.array(airway1),
                       Airway_2_Time2=np.array(airway2),
                       TimeOfDay=np.array(time_of_day),
                       Airway_1=np.array(airway1_code),
                       Airway_2=np.array(airway2_code),
                       ASA=np.array(ASA),
                       if_Emergency=np.array(if_Emergency).astype(int)
                       )

        # reindex
        column_name = ['pid', 'Gender', 'AGE', 'HEIGHT', 'WEIGHT', 'SecondHandSmoke', 'BedName', 'TimeOfDay',
                       'AnesthesiaType', 'ASA', 'if_Emergency', 'ScheduledProcedure', 'Airway_1', 'Airway_1_Time',
                       'Airway_2', 'Airway_2_Time', 'AnesthesiaDuration', 'EBL', 'Urine_Output']
        df = df.reindex(column_name, axis=1)
        df = df.sort_values(by=['pid'], axis=0)

        return df

    def digitalize_val(self, dataframe, column_name):
        """Transform discrete features into coded numbers"""
        type = dataframe[column_name].unique().tolist()
        if np.nan in type:
            type.remove(np.nan)
        mapping = dict(zip(type, list(range(len(type)))))
        mapping[np.nan] = np.nan
        value_list = [mapping[ele] for ele in dataframe[column_name].values]
        return value_list

    def extract_id_converter(self):
        df = self.df_demo
        patient_id = df['PatientID'].tolist()
        pid = list(range(len(df)))
        id_converter = dict(zip(patient_id, pid))

        return id_converter

    def extract_start_time(self):
        df = self.df_demo
        anesthesia_start = df['Anesthesia Start'].tolist()
        patient_ids = df['PatientID'].tolist()
        start_time_dict = dict(zip(patient_ids, anesthesia_start))

        return start_time_dict

    def extract_stop_time(self):
        df = self.df_demo
        anesthesia_stop = df['Anesthesia Stop'].tolist()
        patient_ids = df['PatientID'].tolist()
        stop_time_dict = dict(zip(patient_ids, anesthesia_stop))

        return stop_time_dict

    def time_abs_to_relative(self):
        df = self.df_demo
        birth_days = df['DOB'].tolist()
        anesthesia_start = df['Anesthesia Start'].tolist()
        anesthesia_stop = df['Anesthesia Stop'].tolist()
        airway_time_1 = df['Airway_Event1_Time'].tolist()
        airway_time_2 = df['Airway_Event2_Time'].tolist()

        durations = []
        ages = []
        airway1 = []
        airway2 = []
        time_of_day = []

        for ind, start_time in enumerate(anesthesia_start):
            stop_time = anesthesia_stop[ind]
            birth_time = birth_days[ind]
            # anesthesia duration by minute
            durations.append((datetime.strptime(stop_time, '%m/%d/%y %H:%M')
                              - datetime.strptime(start_time, '%m/%d/%y %H:%M')).seconds / 60)
            # age by year
            try:
                # ages.append((datetime.strptime(start_time, '%m/%d/%y %H:%M')
                #              - datetime.strptime(birth_time, '%m/%d/%y %H:%M')).days / 365)
                ages.append((datetime.strptime(start_time, '%m/%d/%y %H:%M')
                             - datetime.strptime(birth_time, '%m/%d/%y')).days / 365)
            except:
                ages.append(None)

            # airway event time
            try:
                airway1.append((datetime.strptime(airway_time_1[ind], '%m/%d/%y %H:%M')
                                - datetime.strptime(start_time, '%m/%d/%y %H:%M')).seconds / 60)
            except:
                airway1.append(None)
            try:
                airway2.append((datetime.strptime(airway_time_2[ind], '%m/%d/%y %H:%M')
                                - datetime.strptime(start_time, '%m/%d/%y %H:%M')).seconds / 60)
            except:
                airway2.append(None)

            # time of day
            time = start_time.split(' ')[1]
            try:
                time_of_day.append((datetime.strptime(time, '%H:%M') - datetime.strptime('0:0', '%H:%M')).seconds / 60)
            except:
                time_of_day.append(None)

        return durations, ages, airway1, airway2, time_of_day


class ParseDynamicData:

    def __init__(self, vitals_dir):

        self.vitals_dir = vitals_dir

        # real time feature name
        self.type_index_mapping = {'18': 'invDiastolic', '19': 'invMeanBP', '20': 'invSystolic', '23': 'HR',
                                   '25': 'Diastolic', '26': 'MeanBP', '27': 'Systolic', '159': 'SpO2',
                                   '295': 'RespRate', '300': 'PEEP', '627': 'PIP', '728': 'FiO2', '828': 'TidalVolume',
                                   '875': 'Pulse', '1021': 'ETCO2', '1483': 'O2Flow', '1484': 'AirFlow',
                                   '1485': 'N2OFlow', '3292': 'Temp', '3300': 'coreTemp'}
        feat_name = list(self.type_index_mapping.values())
        self.column_name = ['pid', 'ts'] + feat_name
        data_type = [int, int] + [float for count in range(len(feat_name))]
        self.type_dict = dict(zip(self.column_name, data_type))

    def parse_vitals(self, start_time_dict, stop_time_dict, id_converter):

        data_comb = {}
        pid_to_patientID = dict((y, x) for x, y in id_converter.items())
        vitals_files = glob.glob(os.path.join(self.vitals_dir, '*.csv'))

        for file_ind, vitals_file in enumerate(vitals_files):
            print("Parsing file:", file_ind + 1, "/", len(vitals_files), ':', os.path.split(vitals_file)[1])

            with open(vitals_file, 'r') as f:
                lines = f.readlines()

                for line_ind, line in enumerate(lines):
                    if '2020_01' in os.path.split(vitals_file)[1]:
                        if line_ind <= 1:
                            continue
                        feature_list = line.split()
                        if len(feature_list) < 4:
                            continue
                        time = feature_list[2] + ' ' + feature_list[3]
                        value = feature_list[4]
                    else:
                        if line_ind == 0:
                            continue
                        feature_list = line.split(',')
                        if len(feature_list) < 4:
                            continue
                        time = feature_list[2]
                        value = feature_list[3]

                    PatientID = feature_list[0]
                    vital_type = feature_list[1]

                    # convert actual time to relative timestep with surgery starting time as beginning
                    surgery_start_time = start_time_dict[int(PatientID)]
                    surgery_stop_time = stop_time_dict[int(PatientID)]
                    duration = int((datetime.strptime(surgery_stop_time, '%m/%d/%y %H:%M')
                                   - datetime.strptime(surgery_start_time, '%m/%d/%y %H:%M')).seconds / 60)
                    current_time = int((datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
                                       - datetime.strptime(surgery_start_time, '%m/%d/%y %H:%M')).seconds / 60)

                    # only extract measurements within anesthesia procedure
                    if current_time >= duration or current_time < 0:
                        continue

                    # fill data into dictionary
                    if PatientID not in data_comb.keys():
                        data_comb[PatientID] = {}
                    if str(current_time) not in data_comb[PatientID].keys():
                        data_comb[PatientID][str(current_time)] = {}
                    data_comb[PatientID][str(current_time)][self.type_index_mapping[vital_type]] = float(value)

        # Convert dictionary into DataFrame
        rows = []
        for PatientID in data_comb:
            patient_data = data_comb[PatientID]
            for time in patient_data:
                row = patient_data[time]
                row['pid'] = id_converter[int(PatientID)]
                row['ts'] = int(time)
                rows.append(row)
        df = pd.DataFrame(rows)
        df = df.reindex(self.column_name, axis=1)  # sort DataFrame by column_name
        df = df.astype(self.type_dict)
        df.sort_values(by=['pid', 'ts'], axis=0, inplace=True)
        # add index column
        index_arr = np.array(range(len(df)))
        df = df.assign(index=index_arr)
        df = df.reindex(['index'] + self.column_name, axis=1)

        return df

