if [ $# == 0 ]; then
  echo "Please specify model and dataset!"
  exit 2
fi

if [ ${dataset} == "yoochoose1_4" ]; then
  python -u main.py --dataset yoochoose1_4 > logs-srgnn-yoochoose1_4.out
elif [ ${dataset} == "yoochoose1_64" ]; then
  python -u main.py --dataset yoochoose1_64 > logs-srgnn-yoochoose1_64.out
elif [ ${dataset} == "diginetica" ]; then
  python -u main.py --dataset diginetica > logs-srgnn-diginetica.out

else
    echo "(Error) There is no such model or dataset"
    exit 2
fi

echo "Train is started", ${model}, ${dataset}