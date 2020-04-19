#!/usr/bin/env bash
PYTHONPATH="/Users/ericliu/Desktop/Hypoxemia-MLPred/:$PYTHONPATH"
PYTHONPATH="/home/hanyang/project/Hypoxemia-MLPred/:$PYTHONPATH"
export PYTHONPATH

cd ..

python feature_extraction.py --if_impute True --static_txt bow --dynamic_txt notxt
python feature_extraction.py --if_impute True --static_txt rbow-pca --dynamic_txt rbow
python feature_extraction.py --if_impute False --static_txt bow --dynamic_txt notxt
python feature_extraction.py --if_impute False --static_txt rbow --dynamic_txt rbow


