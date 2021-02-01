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
    a = []
    for i in range(len(b)):
        d = b[i].split(' ')
        # print(type(d))
        # print(len(d))
        if len(d) == 1027:
            d = d[1:1025]
            # print(len(d))
            # print(d)
            d = list(map(float, d))
            # print(type(d))
            a.append(d)
    p = pd.DataFrame(a)
    p.to_csv('x.csv',index=False,header=False)

            # print(d)
            # print(type(d))
            # d = np.mat(d)             #转换成矩阵







    #         a.append(d)
    # p = pd.DataFrame(a, columns=None)
    # #             # # print(p)
    # p.to_csv('x.csv', header=False, index=False)

