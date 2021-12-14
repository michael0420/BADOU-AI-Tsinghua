import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def show_gray(img_):
    img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    #cv.imshow('gray', img_)
    #cv.waitKey(100)
    cv.imwrite('lenna_gray.png', img_)
    print('gray over!')
    return img_
def show_binary(img_, approach):
    if approach == 'plt':
        img_ = np.where(img_ >= 128, 1, 0)
        plt.imshow(img_, cmap = 'gray')
        plt.savefig('lenna_binary.png')
        plt.show()
    if approach == 'cv':
        #https://www.cnblogs.com/ssyfj/p/9272615.html
        ret, binary = cv.threshold(img_, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        print("threshold value: %d" % ret)
        #cv.imshow('binary', binary)
        #cv.waitKey(100)
        cv.imwrite('lenna_binary_cv.png', binary)
    print('binary over!')
def nearest_img(img_, h, w):
    height, width, channel = img_.shape
    if height == h and width == w:
        return img_.copy()
    img_copy = np.zeros((h, w, channel), np.uint8)
    sh, sw = h / height, w / width
    for i in range(h):
        for j in range(w):
            x = int(i / sh)
            y = int(j / sw)
            img_copy[i, j] = img[x, y]
    cv.imwrite('lenna_nearest.png', img_copy)
    print('nearest over!')
def bilinear_img(img_, h, w):
    height, width, channel = img_.shape
    print("height: %d, width: %d"% (height, width))
    if height == h and width == w:
        return img_.copy()
    bilinear_img = np.zeros((h, w, 3), dtype=np.uint8)
    sh, sw = height / h, width / w
    for dst_y in range(h):
        for dst_x in range(w):
            src_x = (dst_x + 0.5) * sw - 0.5
            src_y = (dst_y + 0.5) * sh - 0.5
            # (x0, y0)-A-(x1, y0)
            #    |    |     |
            #    |   dst    |
            #    |    |     |
            # (x0, y1)-B-(x1, y1)
            x0 = max(0, int(np.floor(src_x)))
            x1 = min(x0 + 1, width - 1)
            y0 = max(0, int(np.floor(src_y)))
            y1 = min(y0 + 1, height - 1)
            for i in range(3):
                A = (x1 - src_x) * img_[y0, x0, i] + (src_x - x0) * img_[y0, x1, i]
                B = (x1 - src_x) * img_[y1, x0, i] + (src_x - x0) * img_[y1, x1, i]
                bilinear_img[dst_y, dst_x, i] = int((y1 - src_y) * A + (src_y - y0) * B)
    cv.imwrite('lenna_bilinear.png', bilinear_img)
    print('bilinear over!')
if __name__ == '__main__':
    img = cv.imread('lenna.png')
    img_gray = show_gray(img)
    show_binary(img_gray, 'plt')
    show_binary(img_gray, 'cv')
    nearest_img(img, 1024, 1024)
    bilinear_img(img, 1024, 1024)


