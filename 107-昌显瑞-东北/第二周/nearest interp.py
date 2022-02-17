import cv2
import numpy as np

YOUR_WANT = 650

def nearest_interp(img):
    h, w, d = img.shape
    h_rate = YOUR_WANT / h
    w_rete = YOUR_WANT / w
    new_image = np.zeros((YOUR_WANT, YOUR_WANT, d), np.uint8)
    for i in range(YOUR_WANT):
        for j in range(YOUR_WANT):
            x, y = int(i / h_rate), int(j / w_rete)
            new_image[i, j] = img[x, y]
    return new_image

img = cv2.imread("lenna.png")
new_img = nearest_interp(img)

cv2.imshow("img", img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)
