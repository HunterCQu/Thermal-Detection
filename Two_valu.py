import cv2
import matplotlib.pyplot as plt
import os


for filename in os.listdir(r"./" + "n_outmap"):

    img = cv2.imread("n_outmap" + "/" + filename,0)
    ret,img1 = cv2.threshold(img,55,255,cv2.THRESH_BINARY_INV)

    # cv2.imwrite("two_valmap" + "/" + filename,img1)
    cv2.imshow('image',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.xticks([]),plt.yticks([])
# plt.imshow(img1,'gray')
# plt.show()