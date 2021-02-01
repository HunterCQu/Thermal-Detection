import cv2
import os
import numpy as np


array_of_imgx = []
def read_trainx(directory_name):
    for filename in os.listdir(r"./" + directory_name):
        img_x = cv2.imread(directory_name + "/" + filename)
        array_of_imgx.append(img_x)
        # print(img_x.shape)        #(480,640,3)
        # cv2.namedWindow('input_img',cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('input_img',img)
        # cv2.waitKey(0)
        # cv2.destroyWindow()

read_trainx('tmp_save')



array_of_imgy=[]
def read_trainy(directory_name):
    for filename in os.listdir(r'./' + directory_name):
        img_y = cv2.imread(directory_name + "/" + filename)
        array_of_imgy.append(img_y)
        # print(img_y.shape)
        # cv2.namedWindow('input_img',cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('input_img',img_y)
        # cv2.waitKey(0)

# read_trainy('two_valmap')