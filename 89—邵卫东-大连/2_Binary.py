from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
#读取图像
img = Image.open(r"lenna.png")

img = np.array(img)
h, w = img.shape[:2] #获取图像高和宽


# for i in range(h):
#     for j in range(w):
#         m = img[i,j]
#         img_gray[i,j] = int(m[0] * 0.3 + m[1] * 0.59 +m[2] * 0.11)
#
#
# print(img_gray)
# print('image show gray:',img_gray)
# plt.imshow(img_gray,cmap='gray')
# plt.show()

# 灰度化
img_gray = rgb2gray(img)

plt.subplot(121)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
for i in range(h):
    for j in range(w):
        if img_gray[i,j] <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1

img_binary = img_gray
print('image_binary')
print(img_binary)
print(img_binary.shape)
plt.subplot(122)
plt.imshow(img_binary,cmap='gray')
plt.show()