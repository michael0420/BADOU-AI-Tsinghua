import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lenna.png')
img_chans_org = cv2.split(img)
h,w,channels = img.shape
dst_img = np.zeros((h,w,channels),np.uint8)
scale = 256/(h*w)

for i in range(channels):
    hist_org = cv2.calcHist([img_chans_org[i]], [0], None, [256], [0, 256])
    nums_list = [hist_org[0]]
    for p in range(1,256):
        nums_list.append(nums_list[p-1]+hist_org[p])
    for j in range(h):
        for k in range(w):
            dst_img[j,k,i] = int(nums_list[img[j,k,i]]*scale-1)

img_chans_dst = cv2.split(dst_img)

def plot_h(img_channels):
    colors = ('b', 'g', 'r')
    for (img_channel,color) in zip(img_channels,colors):
        hist = cv2.calcHist([img_channel],[0],None,[256],[0,256])
        plt.plot(hist,color=color)

fig = plt.figure()
plt.subplot(1,2,1)
plt.title("Flattened Color Histogram_org")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plot_h(img_chans_org)
plt.subplot(1,2,2)
plt.title("Flattened Color Histogram_dst")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plot_h(img_chans_dst)
plt.show()

cv2.imshow('orgin',img)
cv2.imshow('dst',dst_img)
cv2.waitKey(0)
