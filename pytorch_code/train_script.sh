#!/bin/bash

# How to run
#$ bash train_script.sh simpleGRU yoochoose CrossEntropy 5 50
#$ bash train_script.sh simpleGRU retailrocket CrossEntropy 5 50
#$ bash train_script.sh GRU4Rec yoochoose TOP1-max 5 50
#$ bash train_script.sh GRU4Rec retailrocket TOP1-max 5 50


# How to kill
#
#squeue -u tk2637 -n TimeRec-batch

model=$1
dataset=$2
loss=$3
epoch=$4
batch_size=$5

echo ${model}, ${dataset}, ${loss}, ${epoch}, ${batch_size}

retailrocket_path="--data_folder ../_data/retailrocket-prep --train_data retailrocket-train.csv --valid_data retailrocket-valid.csv --item2idx_dict item_idx_dict_filtered.pkl"
yoochoose_path="--data_folder ../_data/yoochoose-prep/ --train_data yoochoose-train.csv --valid_data yoochoose-valid.csv"

if [ $# == 0 ]; then
  echo "Please specify model and dataset!"
  exit 2
fi

if [ ${dataset} == "yoochoose1_4" ]; then
  python -u main.py --dataset ${dataset} > logs-srgnn-yoochoose1_4.out
elif [ ${dataset} == "yoochoose1_64" ]; then
  python -u main.py --dataset ${dataset} > logs-srgnn-yoochoose1_64.out
elif [ ${dataset} == "diginetica" ]; then
  python -u main.py --dataset ${dataset} > logs-srgnn-diginetica.out

else
    echo "(Error) There is no such model or dataset"
    exit 2
fi

echo "Train is started", ${model}, ${dataset}