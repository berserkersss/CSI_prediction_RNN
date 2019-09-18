#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
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
        #if args.iid:
        dict_users_iid_temp = (mnist_iid(dataset_train, args.num_users))
        #else:
        dict_users = mnist_noniid(dataset_train, args.num_users)
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
        net_glob_fl = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
        net_glob_cl = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob_fl.train()
    net_glob_cl.train()

    # copy weights
    w_glob_fl = net_glob_fl.state_dict()
    w_glob_cl = net_glob_cl.state_dict()


    # training
    eta = 0.01
    Nepoch = 5 # num of epoch 
    loss_train_fl, loss_train_cl = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    para_g = []
    loss_grad = []
    delta_batch_loss_list = []
    beta_list = []
    count_list = np.zeros(256).tolist()
    line1_iter_list = []
    line2_iter_list = []
    wgfed_list = []
    wgcl_list = []

    w_locals, loss_locals = [], []
    w0_locals,loss0_locals =[], []
    weight_div_list = []
    para_cl = []
    para_fl = []
    beta_locals, mu_locals, sigma_locals = [],[],[]
    x_stat_loacals, pxm_locals =[],[]
    data_locals = [[] for i in range(args.epochs)]
    w_fl_iter,w_cl_iter = [], []
    beta_max_his, mu_max_his, sigma_max_his = [], [], []
    acc_train_cl_his, acc_train_fl_his = [], []

    net_glob_fl.eval()
    acc_train_cl, loss_train_clxx = test_img(net_glob_cl, dataset_train, args)
    acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
    acc_train_cl_his.append(acc_test_cl)
    acc_train_fl_his.append(acc_test_cl)
    print("Training accuracy: {:.2f}".format(acc_train_cl))
    print("Testing accuracy: {:.2f}".format(acc_test_cl))


    dict_users_iid = []
    for iter in range(args.num_users):
        dict_users_iid.extend(dict_users_iid_temp[iter])
    # Centralized learning
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        glob_cl = CLUpdate(args=args, dataset=dataset_train, idxs=dict_users_iid)
        w_cl, loss_cl = glob_cl.cltrain(net=copy.deepcopy(net_glob_cl).to(args.device))
        w_cl_iter.append(copy.deepcopy(w_cl))
        net_glob_cl.load_state_dict(w_cl)
        loss_train_cl.append(loss_cl)  # loss of CL
        print('cl,iter = ', iter, 'loss=', loss_cl)

        # testing
        acc_train_cl, loss_train_clxx = test_img(net_glob_cl, dataset_train, args)
        acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train_cl))
        print("Testing accuracy: {:.2f}".format(acc_test_cl))
        acc_train_cl_his.append(acc_test_cl.item())


    # FL setting
    for iter in range(args.epochs): # num of iterations
        w_locals, loss_locals, d_locals  = [], [], []
        beta_locals, mu_locals, sigma_locals = [], [], []
        x_stat_loacals, pxm_locals =[],[]

        # M clients local update
        m = max(int(args.frac * args.num_users), 1) # num of selected users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # select randomly m clients
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # data select
            w, loss, delta_bloss, beta, x_stat, d_local = local.train(net=copy.deepcopy(net_glob_fl).to(args.device))
            x_value, count = np.unique(x_stat,return_counts=True) # compute the P(Xm)
            w_locals.append(copy.deepcopy(w))# collect local model
            loss_locals.append(copy.deepcopy(loss))#collect local loss fucntion
            d_locals.extend(d_local)# collect the isx of local training data in FL

            beta_locals.append(np.max(beta))# beta value
            mu_locals.append(np.max(delta_bloss)) # mu value
            sigma_locals.append(np.std(delta_bloss))#sigma value
            x_stat_loacals.append(x_stat) # Xm
            pxm_locals.append(np.array(count/(np.sum(count)))) #P(Xm)

        data_locals[iter] = d_locals#collect dta
        w_glob_fl = FedAvg(w_locals)# update the global model
        net_glob_fl.load_state_dict(w_glob_fl)# copy weight to net_glob
        w_fl_iter.append(copy.deepcopy(w_glob_fl))

        loss_fl = sum(loss_locals) / len(loss_locals)
        loss_train_fl.append(loss_fl) # loss of FL

        # compute P(Xg)
        xg_value, xg_count = np.unique(x_stat_loacals,return_counts=True)
        xg_count = np.array(xg_count)/(np.sum(xg_count))
        print('fl,iter = ',iter,'loss=',loss_fl)

        # compute beta, mu, sigma
        beta_max = (np.max(beta_locals))
        mu_max = (np.max(mu_locals))
        sigma_max = (np.max(sigma_locals))

        beta_max_his.append(np.max(beta_locals))
        mu_max_his.append(np.max(mu_locals))
        sigma_max_his.append(np.max(sigma_locals))

        # print('beta=', beta_max)
        # print('mu=', mu_max)
        # print('sigma=', sigma_max)

        # testing
        net_glob_fl.eval()
        acc_train_fl, loss_train_flxx = test_img(net_glob_fl, dataset_train, args)
        acc_test_fl, loss_test_flxx = test_img(net_glob_fl, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train_fl))
        print("Testing accuracy: {:.2f}".format(acc_test_fl))

        line1_list=[]
        # the weight divergence of numerical line
        for j in range(len(pxm_locals)):
            lditem1 = sigma_max*(np.sqrt(2/(np.pi*50*(iter+1)))+np.sqrt(2/(np.pi*50*m*(iter+1))))
            lditem2 = mu_max*(np.abs(pxm_locals[j]-xg_count))
            lditem3= 50*(iter+1)*(((1+eta*beta_max)**((iter+1)*Nepoch))-1)/(50*m*(iter+1)*beta_max) # 50 is batch size (10)* num of epoch (5)
            line1 = lditem3*(lditem1+lditem2)
            line1_list.append(line1) # m clients
        line1_iter_list.append(np.sum(line1_list)) # iter elements
        acc_train_fl_his.append(acc_test_fl.item())




    #weight divergence of simulation 
    for i in range(len(w_cl_iter)):
        para_cl = w_cl_iter[i]['layer_input.weight']
        para_fl = w_fl_iter[i]['layer_input.weight']
        line2 = torch.norm(para_cl-para_fl)
        line2_iter_list.append(line2.item())

    print('y_line1=',line1_iter_list)# numerical 
    print('y_line2=',line2_iter_list) # simulation 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(line2_iter_list, c="red")
    plt.xlabel('Iterations')
    plt.ylabel('Difference')
    plt.savefig('Figure/different.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(beta_max_his, c="red")
    plt.xlabel('Iterations')
    plt.ylabel('Beta_max')
    plt.savefig('Figure/beta_max.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sigma_max_his, c="red")
    plt.xlabel('Iterations')
    plt.ylabel('Sigma_max')
    plt.savefig('Figure/sigma_max.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mu_max_his, c="red")
    plt.xlabel('Iterations')
    plt.ylabel('Mu_max')
    plt.savefig('Figure/mu_max.png')


    colors = ["blue", "red"]
    labels = ["non-iid", "iid"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
    ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
    ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Figure/Accuracy_non_iid2_temp.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(line1_iter_list, c=colors[0])
    plt.xlabel('Local_Iterations')
    plt.ylabel('Grad')
    plt.savefig('Figure/numerical _temp.png')



