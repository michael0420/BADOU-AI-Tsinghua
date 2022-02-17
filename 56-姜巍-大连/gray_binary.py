import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

image_src = cv2.imread("lenna.png")

# first put original image in position 131
plt.subplot(341)
img = plt.imread("lenna.png")
plt.imshow(img)
plt.title('original image', fontsize='small')

# get height & width of destination gray image
height, width = image_src.shape[:2]
# initiate a 0-filled grey_image_01 which has the same height & width with image_src, but only 1 channel.
image_01 = np.zeros((height, width), image_src.dtype)
image_02 = np.zeros((height, width), image_src.dtype)
image_03 = np.zeros((height, width), image_src.dtype)
image_04 = np.zeros((height, width), image_src.dtype)
image_05 = np.zeros((height, width), image_src.dtype)
image_06 = np.zeros((height, width), image_src.dtype)
image_07 = np.zeros((height, width), image_src.dtype)
image_08 = np.zeros((height, width, 3), image_src.dtype)
image_09 = np.zeros((height, width, 3))
image_10 = np.zeros((height, width), image_src.dtype)

# write a float algorithm
for y in range(height):
    for x in range(width):
        image_01[y, x] = int(0.11 * image_src[y, x, 0] + 0.59 * image_src[y, x, 1] + 0.3 * image_src[y, x, 2])
plt.subplot(342)
plt.imshow(image_01, cmap='gray')
plt.title('float algorithm', fontsize='small')

# using function of opencv
plt.subplot(343)
image_02 = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
plt.imshow(image_02, cmap='gray')
plt.title('cv2.cvtColor', fontsize='small')

# write a integer algorithm
for y in range(height):
    for x in range(width):
        image_03[y, x] = int((11 * image_src[y, x, 0] + 59 * image_src[y, x, 1] + 30 * image_src[y, x, 2]) / 100)
plt.subplot(344)
plt.imshow(image_03, cmap='gray')
plt.title('integer algorithm', fontsize='small')

# write a average algorithm
for y in range(height):
    for x in range(width):
        image_04[y, x] = int((image_src[y, x, 0] / 3 + image_src[y, x, 1] / 3 + image_src[y, x, 2] / 3))
plt.subplot(345)
plt.imshow(image_04, cmap='gray')
plt.title('average algorithm', fontsize='small')

# write a only get blue
for y in range(height):
    for x in range(width):
        image_05[y, x] = image_src[y, x, 0]
plt.subplot(346)
plt.imshow(image_05)
plt.title('only Blue channel', fontsize='small')

# write a only get blue
for y in range(height):
    for x in range(width):
        image_06[y, x] = image_src[y, x, 1]
plt.subplot(347)
plt.imshow(image_06)
plt.title('only Green channel', fontsize='small')

# write a only get red
for y in range(height):
    for x in range(width):
        image_07[y, x] = image_src[y, x, 2]
plt.subplot(348)
plt.imshow(image_07)
plt.title('only Red channel', fontsize='small')

# write a copy(dtype=uint8)
for i in range(3):
    for y in range(height):
        for x in range(width):
            image_08[y, x, i] = image_src[y, x, i]
plt.subplot(349)
plt.imshow(image_08[:, :, ::-1])
plt.title('forge a copy', fontsize='small')

# write a copy(dtype=float64)
for i in range(3):
    for y in range(height):
        for x in range(width):
            image_09[y, x, i] = image_src[y, x, i] / 255
plt.subplot(3, 4, 10)
plt.imshow(image_09[:, :, ::-1])
plt.title('forge a copy_02', fontsize='small')

# write a binary image
image_10 = np.where(image_01 >= 127, 255, 0)
plt.subplot(3, 4, 11)
plt.imshow(image_10, cmap='gray')
plt.title('binary image', fontsize='small')

plt.show()
