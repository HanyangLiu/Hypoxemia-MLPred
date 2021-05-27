#!/usr/bin/env bash
PYTHONPATH="/Users/ericliu/Desktop/Hypoxemia-MLPred/:$PYTHONPATH"
PYTHONPATH="/home/hanyang/project/Hypoxemia-MLPred/:$PYTHONPATH"
PYTHONPATH="/research-projects/tantra/hanyang/Hypoxemia-MLPred/:$PYTHONPATH"
export PYTHONPATH

train_init() {
    python train_initial_gbtree.py --rs_model $1
#    python train_realtime_gbtree.py --rs_model $1 --feature_file dynamic-ewm-notxt-nonimp.csv --gb_tool catboost --lr 0.02 --depth 6 --l2 1
}

cd ..

for rand in 0 2 3 4 5 6 7 8
do
  train_init ${rand}
done



