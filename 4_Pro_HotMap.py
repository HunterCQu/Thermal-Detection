import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os


# 图片去躁
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./" + directory_name):
        # print(filename) #just for test
        # img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        medina = cv2.medianBlur(gray, 31)
        # print(type(medina))
        # exit()
        # array_of_img.append(medina)
        cv2.imwrite("N_outmap" + "/" + filename, medina[:, :, (2, 1, 0)])
        # print(img)
        # print(array_of_img)
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.imshow(medina)
        # plt.savefig("n_outmap" + "/" + filename)
        # plt.clf()
        # plt.show()
read_directory("H_map")


# 二值化处理
def two_valu(directory_name):
    for filename in os.listdir(r"./" + directory_name):
        img = cv2.imread(directory_name + "/" + filename, 0)
        dim = (640, 640)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        ret, img1 = cv2.threshold(resized_img, 61, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("two_valmap" + "/" + filename, img1)
        # print(img1.shape)
        # cv2.imshow('image',img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
two_valu("N_outmap")