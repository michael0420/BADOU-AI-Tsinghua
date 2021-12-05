import cv2
import numpy as np
def nearest_interpolation(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    after_height=800/height
    after_width=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/after_height)
            y=int(j/after_width)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("lenna.png")
after_image=nearest_interpolation(img)
print(after_image)
print(after_image.shape)
cv2.imshow("nearest interp",after_image)
cv2.imshow("image",img)
cv2.waitKey(0)