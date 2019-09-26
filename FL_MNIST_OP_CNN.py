#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 这个用于CNN的仿真, 手写体只需要一个通道
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import CLUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.Fed import FedAvg_Optimize
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)



    # sample users
    num_img = [1000, 600, 600, 400, 400]
    num_label = [2, 2, 8, 8, 8]
    Ld = [0.0817, 0.1093, 0.2976, 0.2987, 0.2127]

    dict_users = {}
    for k in range(len(num_img)):
        csv_path_train_data = 'csv/' + 'user' + str(k) + 'train_index' + '.csv'

        # 导入MNIST数据集，数据集由 60,000 个训练样本和 10,000 个测试样本组成，每个样本
        # 为一个28*28的图片，读入时我们将这个图片转换为1*784的向量
        # header=None 表示文件一开始就是数据
        train_index = pd.read_csv(csv_path_train_data, header=None)

        # 修剪数据集使得只有图片和标签,把序号剔除
        train_index = train_index.values
        train_index = train_index.T
        dict_users[k] = np.array(train_index[0].astype(int))

    img_size = dataset_train[0][0].shape
    # print('img_size=',img_size)

    net_glob = CNNMnist(args=args).to(args.device)


    net_glob_fl = copy.deepcopy(net_glob)
    net_glob_cl = copy.deepcopy(net_glob)

    net_glob_fl.train()
    net_glob_cl.train()

    # copy weights
    w_glob_fl = net_glob_fl.state_dict()
    w_glob_cl = net_glob_cl.state_dict()

    # training
    loss_train_fl, loss_train_cl, loss_train_iid = [], [], []
    acc_train_cl_his, acc_train_fl_his, acc_train_iid_his = [], [], []

    net_glob_cl.eval()
    acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
    acc_train_cl_his.append(acc_test_cl)
    acc_train_fl_his.append(acc_test_cl)
    print("Testing accuracy: {:.2f}".format(acc_test_cl))

    # FL setting
    for iter in range(args.epochs):  # num of iterations
        w_locals, loss_locals = [], []
        # M clients local update
        m = max(int(args.frac * args.num_users), 1)  # num of selected users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select randomly m clients
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
            w, loss = local.train(net=copy.deepcopy(net_glob_fl).to(args.device))
            w_locals.append(copy.deepcopy(w))  # collect local model
            loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

        w_glob_fl = FedAvg(w_locals)  # update the global model
        net_glob_fl.load_state_dict(w_glob_fl)  # copy weight to net_glob

        loss_fl = sum(loss_locals) / len(loss_locals)
        loss_train_fl.append(loss_fl)  # loss of FL
        print('fl,iter = ', iter, 'loss=', loss_fl)
        # testing
        net_glob_fl.eval()
        acc_test_fl, loss_test_flxx = test_img(net_glob_fl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_fl))
        acc_train_fl_his.append(acc_test_fl.item())

    # FL_Optimize setting

    for iter in range(args.epochs):  # num of iterations
        w_locals, loss_locals = [], []
        # M clients local update
        for idx in range(args.num_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
            w, loss = local.train(net=copy.deepcopy(net_glob_cl).to(args.device))
            w_locals.append(copy.deepcopy(w))  # collect local model
            loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

        w_glob_cl = FedAvg_Optimize(w_locals, Ld)  # update the global model
        net_glob_cl.load_state_dict(w_glob_cl)  # copy weight to net_glob

        loss_cl = sum(loss_locals) / len(loss_locals)
        loss_train_cl.append(loss_cl)  # loss of FL
        print('fl_OP,iter = ', iter, 'loss=', loss_cl)
        # testing
        net_glob_cl.eval()
        acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_cl))
        acc_train_cl_his.append(acc_test_cl.item())

    num_img = [600, 600, 600, 600, 600]
    num_label = [2, 2, 8, 8, 8]
    Ld = [0.0625, 0.1125, 0.2716, 0.3522, 0.2012]

    dict_users = {}
    for k in range(len(num_img)):
        csv_path_train_data = 'csv/' + 'user' + str(k) + 'train_index_balance' + '.csv'

        # 导入MNIST数据集，数据集由 60,000 个训练样本和 10,000 个测试样本组成，每个样本
        # 为一个28*28的图片，读入时我们将这个图片转换为1*784的向量
        # header=None 表示文件一开始就是数据
        train_index = pd.read_csv(csv_path_train_data, header=None)

        # 修剪数据集使得只有图片和标签,把序号剔除
        train_index = train_index.values
        train_index = train_index.T
        dict_users[k] = np.array(train_index[0].astype(int))

    net_glob_fl = copy.deepcopy(net_glob)
    net_glob_cl = copy.deepcopy(net_glob)

    net_glob_fl.train()
    net_glob_cl.train()

    # copy weights
    w_glob_fl = net_glob_fl.state_dict()
    w_glob_cl = net_glob_cl.state_dict()

    # training
    loss_train_fl2, loss_train_cl2 = [], []
    acc_train_cl_his2, acc_train_fl_his2 = [], []

    net_glob_cl.eval()
    acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
    acc_train_cl_his2.append(acc_test_cl)
    acc_train_fl_his2.append(acc_test_cl)
    print("Testing accuracy: {:.2f}".format(acc_test_cl))

    # FL_balance setting
    for iter in range(args.epochs):  # num of iterations
        w_locals, loss_locals = [], []
        # M clients local update
        m = max(int(args.frac * args.num_users), 1)  # num of selected users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select randomly m clients
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
            w, loss = local.train(net=copy.deepcopy(net_glob_fl).to(args.device))
            w_locals.append(copy.deepcopy(w))  # collect local model
            loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

        w_glob_fl = FedAvg(w_locals)  # update the global model
        net_glob_fl.load_state_dict(w_glob_fl)  # copy weight to net_glob

        loss_fl = sum(loss_locals) / len(loss_locals)
        loss_train_fl2.append(loss_fl)  # loss of FL
        print('fl_balance,iter = ', iter, 'loss=', loss_fl)
        # testing
        net_glob_fl.eval()
        acc_test_fl, loss_test_flxx = test_img(net_glob_fl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_fl))
        acc_train_fl_his2.append(acc_test_fl.item())

    # FL_unbalance_Optimize setting
    for iter in range(args.epochs):  # num of iterations
        w_locals, loss_locals = [], []
        # M clients local update
        for idx in range(args.num_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
            w, loss = local.train(net=copy.deepcopy(net_glob_cl).to(args.device))
            w_locals.append(copy.deepcopy(w))  # collect local model
            loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

        w_glob_cl = FedAvg_Optimize(w_locals, Ld)  # update the global model
        net_glob_cl.load_state_dict(w_glob_cl)  # copy weight to net_glob

        loss_cl = sum(loss_locals) / len(loss_locals)
        loss_train_cl2.append(loss_cl)  # loss of FL
        print('fl_OP_balance,iter = ', iter, 'loss=', loss_cl)
        # testing
        net_glob_cl.eval()
        acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_cl))
        acc_train_cl_his2.append(acc_test_cl.item())

    colors = ["navy", "red", "black", "orange"]
    labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
    ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
    ax.plot(acc_train_fl_his2, c=colors[2], label=labels[2])
    ax.plot(acc_train_cl_his2, c=colors[3], label=labels[3])
    ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Figure/Accuracy_CNN.png')

    filename = 'result/' + "Accuracy_FedAvg_unbalance_CNN.csv"
    pd_data = pd.DataFrame(acc_train_fl_his)
    pd_data.to_csv(filename)

    filename = 'result/' + "Accuracy_FedAvg_FedAvg_Optimize_unbalance_CNN.csv"
    pd_data = pd.DataFrame(acc_train_cl_his)
    pd_data.to_csv(filename)

    filename = 'result/' + "Accuracy_FedAvg_balance_CNN.csv"
    pd_data = pd.DataFrame(acc_train_fl_his2)
    pd_data.to_csv(filename)

    filename = 'result/' + "Accuracy_FedAvg_Optimize_balance_CNN.csv"
    pd_data = pd.DataFrame(acc_train_cl_his2)
    pd_data.to_csv(filename)

    filename = 'result/' + "Loss_FedAvg_unbalance_CNN.csv"
    pd_data = pd.DataFrame(loss_train_fl)
    pd_data.to_csv(filename)

    filename = 'result/' + "Loss_FedAvg_FedAvg_Optimize_unbalance_CNN.csv"
    pd_data = pd.DataFrame(loss_train_cl)
    pd_data.to_csv(filename)

    filename = 'result/' + "Loss_FedAvg_balance_CNN.csv"
    pd_data = pd.DataFrame(loss_train_fl2)
    pd_data.to_csv(filename)

    filename = 'result/' + "Loss_FedAvg_Optimize_balance_CNN.csv"
    pd_data = pd.DataFrame(loss_train_cl2)
    pd_data.to_csv(filename)

