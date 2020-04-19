# Hypoxemia-MLPred

## Usage
1. Edit directories in "file_config/data.conf".
2. Make directories in folder "data":
   ```
   cd data
   mkdir raw_data data_frame features model result
   cd ..
   ```
   Then load raw data files into "data/raw_data". Make sure it contains the following three files/folders: demographic.csv, ICD.csv, vitals/2019_06_27_Mon*.csv.
3. Generate DataFrame from raw data
   ```
   python gen_dataframe.py
   ```
4. Extract static and realtime features from generated DataFrame
   ```
   python feature_extraction.py --if_impute True/False
                                --static_txt bow/rbow
                                --dynamic_txt notxt/rbow
   ```
5. Train initial prediction using extracted features
   ```
   python train_initial_predictor.py --hypoxemia_threshhold 90
                                     --hypoxemia_window 10
                                     --prediction_window 5
                                     --static_feature_file FILENAME_OF_FEATURE.csv
   ```
6. Train real-time prediction using extracted features
   ```
   python train_realtime_predictor.py --hypoxemia_threshhold 90
                                      --hypoxemia_window 10
                                      --prediction_window 5
                                      --dynamic_feature_file FILENAME_OF_FEATURE.csv
   ```
