# import opencv and numpy
import cv2
import numpy as np

# write a function takes an image and dimension, then scale it accordingly
def function(img, dimensions):
    height, width, channels = img.shape
    new_height, new_width = dimensions
    new_image = np.zeros((new_height, new_width, channels), np.uint8)
    sh = height/new_height
    sw = width/new_width
    for j in range(new_height):
        for i in range(new_width):
            x = round(i*sw)
            y = round(j*sh)
            new_image[i,j] = img[x,y]
    return new_image

# expand lenna and display
img = cv2.imread("lenna.png")
new_image = function(img,(1000,1000))

print(new_image)
print(new_image.shape)

cv2.imshow("nearest interpolation", new_image)
cv2.imshow("original image", img)

cv2.waitKey(0)
