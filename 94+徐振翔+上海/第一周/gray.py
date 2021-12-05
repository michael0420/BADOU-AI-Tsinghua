import cv2
import numpy as np
import matplotlib.pyplot as plt


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
# convert to gray image
img_gray_1 = np.zeros_like(img[:, :, 0])
# gray image 1
for i in range(h):
    for j in range(w):
        img_gray_1[i, j] = 0.11 * img[i, j, 0] + 0.59 * img[i, j, 1] + 0.3 * img[i, j, 1]
# gray image 2
img_gray_2 = cv2.imread(".\\lenna.png", 0)
# gray image 3
img_gray_3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray image 4 -- 3 channel
img_gray_4 = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        img_gray_4[i, j, :] = img_gray_1[i, j]

# merge image
img_out1 = np.hstack((img, img_gray_4))
img_gray = np.hstack((img_gray_1, img_gray_2, img_gray_3))
# show source image
cv_show("out", img_out1)
cv_show("gray", img_gray)
# another function to show image
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.imshow(img_gray_1, cmap='gray')
plt.subplot(223)
plt.imshow(img_gray_2, cmap='gray')
plt.subplot(224)
plt.imshow(img_gray_3, cmap='gray')
plt.show()

# Binary image
img_bin_1 = np.zeros_like(img_gray_1)
# Binary image 1
for i in range(h):
    for j in range(w):
        if img_gray_1[i, j] <= 80:
            img_bin_1[i, j] = 0
        else:
            img_bin_1[i, j] = 255
# Binary image 2
ret, img_bin_2 = cv2.threshold(img_gray_1, 120, 255, cv2.THRESH_BINARY)
# Binary image 3
img_bin_3 = np.where(img_gray_1 >= 160, 255, 0)
# merge image
cv_bin = np.hstack((img_bin_1, img_bin_2, img_bin_3))
cv_bin = np.array(cv_bin, dtype=int)
# show image
cv_bin_0 = np.zeros((h, w * 3, 3))
for i in range(h):
    for j in range(w * 3):
        cv_bin_0[i, j, :] = cv_bin[i, j]
cv_show("img", cv_bin_0)

plt.subplot(111)
plt.imshow(cv_bin, cmap='gray')
plt.show()
