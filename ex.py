import numpy as np
import seaborn as sns
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2

with open('1.txt', 'r') as f:
    data = f.read()
    # data = data.replace('\n','')
    b = data.split('endstart')

    # print(b)
    # print(len(b))
    for i in range(len(b)):
        d = b[i].split(' ')
        # print(type(d))
        # print(len(d))
        if len(d) == 1027:
            d = d[1:1025]
            # print(len(d))
            # print(d)
            d = list(map(float, d))
            # d = np.mat(d)                       #变成矩阵
            # print(type(d))
            # d = np.reshape(d, (32, 32))
            # print(d.shape)
            # h_map = sns.heatmap(d, annot=False, linewidths=0, linecolor='white', cbar=False,
            #             square=False, xticklabels='', yticklabels='')

            # plt.pause(0.1)
            # plt.waitforbuttonpress(0)
            # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            # h_map = plt.figure(figsize=(5.12,5.12))
            # plt.savefig('tmp_save/H_map_{}'.format(i))
            # print(type(h_map))
            # plt.show()

