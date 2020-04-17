#!/usr/bin/env bash
PYTHONPATH="/Users/ericliu/Desktop/Hypoxemia-MLPred/:$PYTHONPATH"
PYTHONPATH="/home/hanyang/project/Hypoxemia-MLPred/:$PYTHONPATH"
export PYTHONPATH

hypoxemia_window=10
if_impute=True

python gen_dataframe.py
python feature_extraction.py --if_impute ${if_impute}
python train_realtime_predictor.py --hypoxemia_window ${hypoxemia_window}

