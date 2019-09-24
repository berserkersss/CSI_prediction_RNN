#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # sample users

        num_img = [800, 800, 600, 400, 400]
        num_label = [2, 2, 2, 4, 8]
        Ld = [0.1089,   0.1279,    0.1854,    0.1853,    0.3924]
        #Ld = [0.02,   0.02,    0.02,    0.04,    0.9]

        dict_usersx = mnist_noniid(dataset_train, args.num_users)
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

        dict_users_iid_temp = mnist_iid(dataset_train, args.num_users)




    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    #print('img_size=',img_size)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)

    else:
        exit('Error: unrecognized model')

    net_glob_fl = copy.deepcopy(net_glob)
    net_glob_cl = copy.deepcopy(net_glob)
    net_glob_iid = copy.deepcopy(net_glob)
    net_glob_iid.train()
    net_glob_fl.train()
    net_glob_cl.train()

    # copy weights
    w_glob_fl = net_glob_fl.state_dict()
    w_glob_cl = net_glob_cl.state_dict()
    w_glob_iid = net_glob_iid.state_dict()


    # training
    loss_train_fl, loss_train_cl, loss_train_iid = [], [], []
    acc_train_cl_his, acc_train_fl_his, acc_train_iid_his = [], [], []

    net_glob_fl.eval()
    acc_train_cl, loss_train_clxx = test_img(net_glob_cl, dataset_train, args)
    acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
    acc_train_cl_his.append(acc_test_cl)
    acc_train_fl_his.append(acc_test_cl)
    print("Training accuracy: {:.2f}".format(acc_train_cl))
    print("Testing accuracy: {:.2f}".format(acc_test_cl))



    # FL setting
    for iter in range(args.epochs): # num of iterations
        w_locals, loss_locals= [], []
        # M clients local update
        m = max(int(args.frac * args.num_users), 1) # num of selected users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # select randomly m clients
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # data select
            w, loss= local.train(net=copy.deepcopy(net_glob_fl).to(args.device))
            w_locals.append(copy.deepcopy(w))# collect local model
            loss_locals.append(copy.deepcopy(loss))#collect local loss fucntion

        w_glob_fl = FedAvg(w_locals)# update the global model
        net_glob_fl.load_state_dict(w_glob_fl)# copy weight to net_glob


        loss_fl = sum(loss_locals) / len(loss_locals)
        loss_train_fl.append(loss_fl) # loss of FL
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

    # CL_Optimize setting
    dict_user_iid = []
    for iter in range(args.num_users):
        dict_user_iid.extend(dict_users_iid_temp[iter])
    for iter in range(args.epochs):  # num of iterations
        local = CLUpdate(args=args, dataset=dataset_train, idxs=dict_user_iid)  # data select
        w, loss = local.cltrain(net=copy.deepcopy(net_glob_iid).to(args.device))
        net_glob_iid.load_state_dict(w)  # copy weight to net_glob
        print('cl,iter = ', iter, 'loss=', loss)
         # testing
        net_glob_iid.eval()
        acc_test_iid, loss_test_iid = test_img(net_glob_iid, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test_iid))
        acc_train_iid_his.append(acc_test_iid.item())

    colors = ["blue", "red", "black"]
    labels = ["FedAvg", "FedAvg_Optimize", "CL"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
    ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
    ax.plot(acc_train_iid_his, c=colors[2], label=labels[2])
    ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Figure/Accuracy_non_iid2_temp.png')

    filename = "Accuracy_FedAvg.csv"
    pd_data = pd.DataFrame(acc_train_fl_his)
    pd_data.to_csv(filename)

    filename = "Accuracy_FedAvg_Optmize.csv"
    pd_data = pd.DataFrame(acc_train_cl_his)
    pd_data.to_csv(filename)

    filename = "Accuracy_CL.csv"
    pd_data = pd.DataFrame(acc_train_iid_his)
    pd_data.to_csv(filename)




