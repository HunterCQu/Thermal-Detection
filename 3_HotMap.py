import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

data = pd.read_csv('train_X.csv',header=None)
# print(data.shape)      #(6607,1024)
data = data.values
x = data[:,:]
# print(x.shape)
c = 0
for row in x:
    # print(row)
    # print(row.shape)
    # print(type(row))
    c = c+1
    a = np.mat(row)
    a = np.reshape(a,(32,32))
    s = sns.heatmap(a, annot=False, linewidths=0, linecolor='white', cbar=False,
                    square=False, xticklabels='', yticklabels='')  # cbar为F可不显示温度值

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig('H_map/h_map{}'.format(c))
    print(c)