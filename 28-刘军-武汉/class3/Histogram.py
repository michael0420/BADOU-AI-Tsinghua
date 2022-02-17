import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png')
colors = ('b','g','r')
chans = cv2.split(img)
for (chan,color) in zip(chans,colors):
    plt.hist(chan.ravel(),256,color=color)
plt.show()
