# -*- coding:utf-8 -*-
import numpy as np
import cv2

# path = r'C:\Users\高力合\Desktop\project\H_map_4.png'
img = cv2.imread('two_imgimg.png',0)
img0 = img.copy()
print(img.shape)
for w,i in enumerate(img[:-1,:-1]):
    for h,j in enumerate(i):
        if (img[w-1,h]==0 and img[w+1,h]==0) and (img[w,h-1]==0 and img[w,h+1]==0):#or (img[w,h-1] != img[w,h+1])
            img0[w,h]=255
cv2.imshow("t",img0)
cv2.waitKey(0)
cv2.destroyAllWindows()