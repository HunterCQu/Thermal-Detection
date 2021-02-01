import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def process_x():
# 读取数据(温度值：402*1024)-->(402,32,32)
    data_x = pd.read_csv('train_X.csv',header=None,nrows=402)
    # print(data_x)
    data_x = data_x.values
    train_x = data_x.reshape((402,32,32,1))
# print(data_x.shape)
    return train_x


def process_y():
    train_y = []
    for filename in os.listdir(r"./" + "two_valmap"):
        img = cv2.imread("two_valmap" + "/" + filename)
        train_y.append(img)
    # print(train_y)
    train_y = np.array(train_y)
    train_y = train_y / 255
    # print(train_y)
    # print(type(train_y))
    # print(train_y.shape)     #(402,512,512,3)
        # print(img.shape)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return train_y

tx = process_x()
ty = process_y()


network = Sequential([
    layers.Conv2D(128, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation='relu'),
    layers.Conv2D(32, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(32, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(32, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(32, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(16, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(8, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation='relu'),
    layers.Conv2DTranspose(3, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation='relu'),
])
network.build(input_shape=(None,32,32,1))
network.summary()



network.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy']
                )


his = network.fit(tx, ty, epochs=100,batch_size=20, verbose=1,
            validation_freq=5)





z = tf.reshape(tx[:9], (9, 32, 32, 1))
c = network(z)
c = tf.squeeze(c).numpy()
# c = np.where(c > 0.9, 1, 0)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(c[i],vmin=0,vmax=255)
plt.show()



his = his.history
loss = his['loss']
# val_loss = his['val_loss']
aca = his['accuracy']
# acav = his['val_accuracy']

plt.subplot(121)  # 1行 2列 第一张图
plt.plot(loss, label='loss')
# plt.plot(val_loss, label='val_loss')
plt.legend()

plt.subplot(122)  # 1行 2列 第二张图
plt.plot(aca, label='accuracy')
# plt.plot(acav, label='val_accuracy')
plt.legend()
plt.show()