
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('h_map1.png')
cv2.namedWindow('imagshow',2)
cv2.imshow('imagshow',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.hist(gray.ravel(),256,[0,256])
plt.show()