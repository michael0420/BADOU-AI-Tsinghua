import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt直方图
plt.figure(figsize=(6,6))
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.subplot(311)
plt.hist(gray.ravel(), 256)
plt.title("plt hist")
# cv2直方图
hist = cv2.calcHist([gray], [0], None, [256], [0,255])
plt.subplot(312)
plt.plot(hist)
plt.title("cv2 calcHist")
# BGR直方图
channels = cv2.split(img)
colors = ["b", "g", "r"]
plt.subplot(313)
plt.title("3 channels")
for i in range(3):
    series = cv2.calcHist([channels[i]], [0], None, [256], [0,255])
    plt.plot(series, c=colors[i])
plt.show()

# 灰色直方图均衡化
plt.figure(figsize=(13,10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
grayHist = cv2.equalizeHist(gray)
print(grayHist)
plt.subplot(221)
plt.hist(gray.ravel(), 256)
plt.subplot(222)
plt.hist(grayHist.ravel(), 256)
plt.subplot(212)
plt.imshow(np.hstack([gray, grayHist]), cmap="gray")
plt.show()
cv2.imshow("gray Hist", np.hstack([gray, grayHist]))
# 彩色直方图均衡化
plt.figure(figsize=(13, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.subplot(221)
for i in range(3):
    series = cv2.calcHist([channels[i]], [0], None, [256], [0,255])
    plt.plot(series, c=colors[i])
bgrHist = []
series = []
for i in range(3):
    bgrHist.append(cv2.equalizeHist(channels[i]))
    series.append(cv2.calcHist([bgrHist[i]], [0], None, [256], [0,255]))
    plt.subplot(2, 2, i + 2)
    plt.plot(series[i], c=colors[i])
plt.show()
colorHist = cv2.merge(bgrHist)
cv2.imshow("color Hist", np.hstack([img, colorHist]))
cv2.waitKey()

