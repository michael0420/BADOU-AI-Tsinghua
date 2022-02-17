from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
##二值化
img = cv2.imread("/Users/oh/Desktop/zuoye/WechatIMG13.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w =img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j] <= 0.5):
            img_gray[i,j] =0
        else:
            img_gray[i,j] = 1

img_binary = np.where(img_gray <= 0.5,1,0)
cv2.imshow("lena",img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("-----img_binary")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
