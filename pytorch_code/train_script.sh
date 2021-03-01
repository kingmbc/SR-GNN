#!/bin/bash

# How to run
#$ bash train_script.sh SRGNN diginetica
#$ bash train_script.sh SRGNN yoochoose1_4
#$ bash train_script.sh SRGNN yoochoose1_64

# How to kill
#

model=$1
dataset=$2
loss=$3
epoch=$4
batch_size=$5

echo ${model}, ${dataset}, ${loss}, ${epoch}, ${batch_size}

retailrocket_path="--data_folder ../../_data/retailrocket-prep --train_data retailrocket-train.csv --valid_data retailrocket-valid.csv --item2idx_dict item_idx_dict_filtered.pkl"
yoochoose_path="--data_folder ../../_data/yoochoose-prep/ --train_data yoochoose-train.csv --valid_data yoochoose-valid.csv"

yoochoose1_4_path="--data_folder ../../_data/yoochoose-prep/ --train_data yoochoose1_4-train.txt --valid_data yoochoose1_4-test.txt"
yoochoose1_64_path="--data_folder ../../_data/yoochoose-prep/ --train_data yoochoose1_64-train.txt --valid_data yoochoose1_64-test.txt"
diginetica_path="--data_folder ../../_data/diginetica-prep/ --train_data train.txt --valid_data test.txt"

if [ $# == 0 ]; then
  echo "Please specify model and dataset!"
  exit 2
fi

if [ ${dataset} == "yoochoose1_4" ]; then
  python -u main.py --dataset ${dataset} ${yoochoose1_4_path} --batch_size ${batch_size}
elif [ ${dataset} == "yoochoose1_64" ]; then
  python -u main.py --dataset ${dataset} ${yoochoose1_64_path} --batch_size ${batch_size}
elif [ ${dataset} == "diginetica" ]; then
  python -u main.py --dataset ${dataset} ${diginetica_path} --batch_size ${batch_size}

else
    echo "(Error) There is no such model or dataset"
    exit 2
fi

echo "Train is started", ${model}, ${dataset}