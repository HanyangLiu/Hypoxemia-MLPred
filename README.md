# Hypoxemia-MLPred

## Usage
1. Edit directories in "file_config/data.conf".
2. Make directories in folder "data", and then load raw data files into "data/raw_data".
   ```
   cd data
   mkdir raw_data data_frame features model result
   cd ..
   ```
3. Generate DataFrame from raw data
   ```
   python gen_dataframe.py
   ```
4. Extract static and realtime features from generated DataFrame
   ```
   python feature_extraction.py --if_impute True/False
   ```
5. Train initial prediction using extracted features
   ```
   python train_initial_predictor.py --hypoxemia_threshhold 90
                                     --hypoxemia_window 10
                                     --prediction_window 5
   ```
6. Train dynamic prediction using extracted features
   ```
   python train_dynamic_predictor.py --hypoxemia_threshhold 90
                                     --hypoxemia_window 10
                                     --prediction_window 5
   ```
