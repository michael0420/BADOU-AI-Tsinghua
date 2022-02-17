from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#灰度值
img=cv2.imread("/Users/oh/Desktop/zuoye/WechatIMG13.jpeg")
hegiht,width=img.shape[:2]
img_gray = np.zeros([hegiht,width],img.dtype)
for i in range(hegiht):
    for j in range(width):
        m =img[i,j]
        img_gray[i,j]=int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(221)
# img = plt.imread("/Users/oh/Desktop/zuoye/WechatIMG13.jpeg")
# # img = cv2.imread("lenna.png", False)
# plt.imshow(img)
# print("---image lenna----")
# print(img)




