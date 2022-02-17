import cv2
import numpy
import numpy as np


# single channel histogram
def his(src):
    h, w = src.shape
    pixN = dict()
    for i in range(h):
        for j in range(w):
            if(src[i, j] in pixN):
                pixN[src[i, j]] = pixN[src[i, j]] + 1
            else:
                pixN[src[i, j]] = 1
    ind = sorted(pixN)
    sum = 0
    for key in ind:
        pixN[key] += sum
        sum = pixN[key]
    dest = np.zeros([h, w], src.dtype)
    for i in range(h):
        for j in range(w):
            dest[i, j] = int(float(pixN[src[i, j]])/h/w*256-1)
    return dest


img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = his(gray)
print(a)
cv2.imshow("gray", np.hstack([gray, a]))
b, g, r = cv2.split(img)
bh = his(b)
gh = his(g)
rh = his(r)
cv2.imshow("color", np.hstack([img, cv2.merge([bh, gh, rh])]))
cv2.waitKey()
