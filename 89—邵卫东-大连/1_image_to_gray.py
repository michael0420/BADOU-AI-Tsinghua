from PIL import Image
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

#读取图像
img = Image.open(r"lenna.png")


img = np.array(img)
h, w = img.shape[:2] #获取图像高和宽
img_gray = np.zeros([h,w],img.dtype)   #  #创建一张和当前图片大小一样的单通道图片

for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0] * 0.3 + m[1] * 0.59 +m[2] * 0.11)


print(img_gray)
print('image show gray:',img_gray)
plt.imshow(img_gray,cmap='gray')
plt.show()

plt.subplot(121)
img = plt.imread(r"lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(122)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
plt.show()











