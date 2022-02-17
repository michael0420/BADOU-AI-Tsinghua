from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import config


# 灰度化
# red * 0.3 + green * 0.59 + blue * 0.11
# values of gray is in [0,255]; values of black-white contains only two values,eg 0 or 1
# pic read by opencv is BGR, not RGB
# new an empty of origin sized array, transform origin into single gray tunnel

# function 1
readImage = cv2.cvtColor(cv2.imread(config.lena_path),cv2.COLOR_BGR2RGB)
print(readImage)
print(readImage.shape)
print("====")
height,width,channel = readImage.shape
gray = np.zeros([height,width],readImage.dtype)
for i in range(height):
    for j in range(width):
        originRGB = readImage[i,j]
        print(originRGB)
        # origin RGB (3 channels changes to single channel)
        gray[i,j] = int(originRGB[0]*0.3 + originRGB[1]*0.59+originRGB[2]*0.11)
cv2.namedWindow('self_gray', cv2.WINDOW_NORMAL)
cv2.imshow("self_gray",gray)
cv2.waitKey(221)
cv2.imwrite(config.test_out+"lena_gray_f1.jpg",gray)

# function 2
grayImage = cv2.cvtColor(cv2.imread(config.lena_path),cv2.COLOR_BGR2GRAY)
print(grayImage.shape)
cv2.imshow("no_self_gray",grayImage)
cv2.waitKey(221)
cv2.imwrite(config.test_out+"lena_gray_f2.jpg",grayImage)

# function 3 => white-black pic
pltImg = plt.imread(config.lena_path)
grayImage = rgb2gray(pltImg)
print("gray-plt")
print(grayImage)
height,width = grayImage.shape
blackImage = np.zeros([height,width],pltImg.dtype)
for i in range(height):
    for j in range(width):
        if (grayImage[i, j] <= 0.5):
            blackImage[i, j] = 0
        else:
            blackImage[i, j] = 1
plt.subplot(223)
plt.imshow(blackImage, cmap='gray')
plt.show()
plt.imsave(config.test_out+"lena_gray_f3.jpg",blackImage,cmap='gray')


