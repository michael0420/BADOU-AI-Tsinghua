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


def bilinear_interpolation(img_s, img_shape, centeral_symmetry=1):
    h, w, c = np.shape(img_s)
    h1, w1 = img_shape[0], img_shape[1]

    img_d = np.zeros((h1, w1, c), dtype=np.uint8)
    h_s = h/h1
    w_s = w/w1
    # print(h_s, w_s).
    for x0 in range(w1):
        for y0 in range(h1):
            for c0 in range(c):
                if centeral_symmetry == 1:
                    # central symmetry
                    # (x+0.5) = (x0+0.5) * w_src/w_dst
                    x = (x0 + 0.5) * w_s - 0.5
                    y = (y0 + 0.5) * h_s - 0.5
                else:
                    # without central symmetry
                    # (x+0.5) = (x0+0.5) * M/N
                    x = (x0 + 0.5) * w_s - 0.5
                    y = (y0 + 0.5) * h_s - 0.5

                # get 4 boundary point
                x1 = int(x)
                x2 = min(x1 + 1, w - 1)
                y1 = int(y)
                y2 = min(y1 + 1, h - 1)
                # print(x0, y0)
                # print(x1, x2, y1, y2)
                # calculate the interpolation
                # fr1 = (x2-x)*f(Q11)/(1) + (x-x1)*f(Q21)/(1)
                # fr2 = (x2-x)*f(Q12)/(1) + (x-x1)*f(Q22)/(1)
                # f   = (y2-y)*fr1   /(1) + (y-y1)*fr2   /(1)
                fr1 = (x2 - x) * img[y1, x1, c0] + (x - x1) * img[y1, x2, c0]
                fr2 = (x2 - x) * img[y2, x1, c0] + (x - x1) * img[y2, x2, c0]
                img_d[y0, x0, c0] = int((y2 - y) * fr1 + (y - y1) * fr2)
    return img_d


# input image file
img = cv2.imread(".\\lenna.png")
# show source image
cv_show("img", img)
img1 = bilinear_interpolation(img, (1000, 800))
img2 = bilinear_interpolation(img, (1000, 800), 0)
img_out = np.hstack((img1, img2))
# show nearest image
cv_show("img_nearest", img_out)
