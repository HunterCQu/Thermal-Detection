import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import os


array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        medina = cv2.medianBlur(gray, 31)
        # print(type(medina))
        # exit()
        # array_of_img.append(medina)

        cv2.imwrite("n_outmap" + "/" + filename, medina[:,:,(2,1,0)])
        # print(img)
        # print(array_of_img)
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.imshow(medina)
        # plt.savefig("n_outmap" + "/" + filename)
        # plt.clf()
        # plt.show()

read_directory("tmp_save")

# print(array_of_img)

# img = cv2.imread('h_map.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# medina = cv2.medianBlur(gray, 31)
# # print(type(medina))
# # print(type(img))
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.imshow(medina)
# plt.show()
#
#




















# image = cv2.imread('h_map.jpg')
#
# # print(image.shape)       #(480,640,3)
# # print(image)
# # cv2.imshow("image",image)
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# medina = cv2.medianBlur(gray, 41)
#
#
# plt.imshow(medina)
# plt.show()





