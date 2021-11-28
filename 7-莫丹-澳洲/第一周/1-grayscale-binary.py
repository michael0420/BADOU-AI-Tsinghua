# import rgb to grayscale tool (scikit image)
from skimage.color import rgb2gray
# import number crunching tool
import numpy as np
# import ploting tool
import matplotlib.pyplot as plt
# import image reading tools
from PIL import Image
import cv2

### RGB to Gray data manual calculation ###
# 1 read image
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
# create grayscale image
img_gray = np.zeros([h,w],img.dtype)
# 2 loop through pixel and calculate
for i in range(h):
    for j in range(w):
        m = img[i,j]
        # calculate grayscale value from BGR
        img_gray[i,j] = int(m[0]*0.114 + m[1]*0.587 + m[2]*0.229)

#print (img_gray.dtype)
#print("image show gray: %s"%img_gray)
#cv2.imshow("image show gray",img_gray)
#cv2.waitKey(0)

# 1 show image with matplotlib
plt.subplot(2,2,1)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 2 gray-scale
img_gray_2 = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(2,2,2)
plt.imshow(img_gray_2, cmap="gray")
print("---image gray 2----")
print(img_gray_2)

# 3 scale-hot
plt.subplot(2,2,3)
plt.imshow(img_gray, cmap="hot")
print("---image hotter----")
print(img_gray)

# 4 binary
img_binary = np.where(img_gray/255>=0.5, 1, 0)
#print("-----imge_binary------")
#print(img_binary)

plt.subplot(2,2,4)
plt.imshow(img_binary, cmap="gray")

plt.show()