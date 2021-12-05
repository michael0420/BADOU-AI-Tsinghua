import cv2
import numpy as np


def cv_show(name, im):
    if (type(name) != str) or (type(im) != np.ndarray):
        return
    # show image
    cv2.imshow(name, im)
    # wait a key to destroy window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# input image file
img = cv2.imread(".\\lenna.png")
# show source image
cv_show("img", img)
h, w, c = np.shape(img)
# create nearest image size
h1 = 1000
w1 = 1000
img_near = np.zeros((h1, w1, c), dtype=np.uint8)
h_s = h1 / h
w_s = w1 / w
# create nearest image
for i in range(h1):
    for j in range(w1):
        i1 = int((i/h_s)+0.5)
        j1 = int((j/w_s)+0.5)
        img_near[i, j] = img[i1, j1]
# show nearest image
cv_show("img_nearest", img_near)
