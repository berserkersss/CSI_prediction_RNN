#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 这个用于CNN的仿真, 手写体只需要一个通道
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    num = 4
    colors = ["navy", "red", "black", "orange"]
    labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance", "FedAvg_balance", "FedAvg_Optimize_balance"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  导入unbalance数据集
    for i in range(num):
        csv_path_accuracy = 'result/CNN/' + 'Accuracy_' + labels[i] + '_CNN.csv'
        accuracy = pd.read_csv(csv_path_accuracy, header=None)
        accuracy = accuracy.values
        ax.plot(accuracy[0], c=colors[i], label=labels[i])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figure/Accuracy_CNN.png')
