
#nearest interpolation implementation

# import library
import cv2
import numpy as np

# nearest interpolation function

def NearInter(img,dst_height,dst_width):
    height, width, channels = img.shape
    NewImg = np.zeros((dst_height, dst_width, channels), np.uint8)
    step_height = dst_height/height
    step_width = dst_width/width
    for i in range(dst_height):
        for j in range(dst_width):
            x = int(i/step_height)
            y = int(j/step_width)
            NewImg[i,j] = img[x,y]
    return NewImg

img = cv2.imread("AWACS.jpeg")
zoom = NearInter(img,642,960)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.waitKey(2000)
cv2.imshow("image",img)
cv2.waitKey(2000)