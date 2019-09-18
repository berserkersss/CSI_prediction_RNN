#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        img_idx = self.idxs[item]
        return image, label, img_idx


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        img_train_list = []

        dgrad_bl = []
        all_x_list = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss_grad = []
            weight_grad_list = []
            beta_list = []
            last_log = 0
            for batch_idx, (images, labels, img_idxs) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                img_train_list.extend(np.array(img_idxs).tolist()) # collect the idx of FL local training dataset 

                for j in range(10): # this function make x ~ [0,255]
                    x = np.reshape(images.data[j],28*28)
                    x = x.numpy()
                    x = np.round(255*(x*0.3081+0.1307))
                    x = x.tolist()
                    all_x_list.append(x)

                net.zero_grad()
                log_probs = net(images)# predicted label 
                loss = self.loss_func(log_probs, labels) 
                loss.retain_grad() # compute the gradient of batch loss 
                log_probs.retain_grad() # compute the gradienr of loss 
                #loss.register_hook(lambda grad: print(torch.norm(grad)))
                #log_probs.register_hook(lambda grad: print(torch.norm(grad)))
                
                loss.backward()
                optimizer.step()

                next_log = torch.norm(log_probs.grad).numpy().item()
                temp5 = next_log-last_log
                last_log = next_log


                temp1 = net.layer_input.weight.grad
                temp2 = net.layer_hidden.weight.grad
                log_probs.grad.shape
                log_probs.grad.numpy()
                temp1.shape
                temp2.shape

                temp3 = torch.norm(temp1)+torch.norm(temp2) # compute delta w
                temp4 = (temp5)/(temp3.numpy().item()) # compute the beta
                #temp6 = temp5 *

                batch_loss_grad.append(torch.norm(log_probs.grad).numpy().item()) # compute the grad of bacth loss 
                beta_list.append(temp4)# beta list 
                

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), batch_loss_grad, beta_list, all_x_list, img_train_list

class CLUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cl_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=int(len(DatasetSplit(dataset, idxs))/args.num_users), shuffle=True)
        #self.cl_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def cltrain(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss_g = []


        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, img_idxs) in enumerate(self.cl_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #print(images.shape)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                w_g = net.state_dict()
                if self.args.verbose and batch_idx % 60 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss_g.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss_g) / len(epoch_loss_g)

