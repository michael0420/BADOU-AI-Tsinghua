import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_interp(source_image,new_h,new_w):
    deal_image=cv2.imread(source_image)
    new_image=np.zeros((new_h,new_w,3),deal_image.dtype)
    scale_h=new_h/deal_image.shape[0]
    scale_w=new_w/deal_image.shape[1]
    for i in range(new_h):
        for j in range(new_w):
            x=int(i/scale_h)
            y=int(j/scale_w)
            new_image[i,j]=deal_image[x,y]
    return new_image

new_image=nearest_interp("lenna.png",800,800)

plt.subplot(221)
img_show1=plt.imread("lenna.png")
plt.imshow(img_show1)
print("---source image---")
print(img_show1)

plt.subplot(222)
img_show2=cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
plt.imshow(img_show2,cmap="gray")
print("---gray image---")
print(img_show2)
plt.show()
cv2.imshow("new image",new_image)
cv2.waitKey(10000)