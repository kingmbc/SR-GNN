#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')

parser.add_argument('--data_folder', default='../_data/yoochoose-prep/', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)

args = parser.parse_args()
print(args)

wandb.init(project="SR-GNN Project",
           name=args.dataset)
config = wandb.config


def main():
    train_data = pickle.load(open('../../_data/' + args.dataset + '/train.txt', 'rb'))
    if args.validation:
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../../_data/' + args.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../../_data/' + args.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if args.dataset == 'diginetica':
        n_node = 43098
    elif args.dataset == 'yoochoose1_64' or args.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(args, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(epoch, model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print(f'\tRecall@20:\t{best_result[0]:.4f}'
              f'\tMMR@20:\t{best_result[1]:.4f}'
              f'\tEpoch:\t{best_epoch[0]},\t{best_epoch[1]}')
        wandb.log({"best_recall": best_result[0],
                   "best_mrr": best_result[1],
                   "best_recall_epoch": best_epoch[0],
                   "best_mrr_epoch": best_epoch[1]})
        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
