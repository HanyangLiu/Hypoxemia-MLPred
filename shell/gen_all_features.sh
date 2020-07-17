#!/usr/bin/env bash
PYTHONPATH="/Users/ericliu/Desktop/Hypoxemia-MLPred/:$PYTHONPATH"
PYTHONPATH="/home/hanyang/project/Hypoxemia-MLPred/:$PYTHONPATH"
export PYTHONPATH

cd ..

#python feature_extraction.py --type static --static_txt bow
#python feature_extraction.py --type static --static_txt rbow

python feature_extraction.py --type dynamic-ewm --if_impute True --dynamic_txt notxt
#python feature_extraction.py --type dynamic-ewm --if_impute True --dynamic_txt rbow
python feature_extraction.py --type dynamic-ewm --if_impute False --dynamic_txt notxt
#python feature_extraction.py --type dynamic-ewm --if_impute False --dynamic_txt rbow

python feature_extraction.py --type dynamic-sta
