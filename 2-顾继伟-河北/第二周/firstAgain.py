import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv2toGray1():
    # using cv2's imread(src,0) method
    img = cv2.imread("lenna.png", 0)
    cv2.imshow("lenna", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def cv2toGray2():
    # using cv2.cvtColor(src,ENUMcode)
    img = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("lenna", img_gray)
    cv2.waitKey()
    cv2.destroyAllWindows()
def cv2toGray3():
    # using  one of image's RGB as Gray-image
    img = cv2.imread("lenna.png", 1)
    b, g, r = cv2.split(img)
    cv2.imshow("testb", b)
    cv2.waitKey(0)
    cv2.imshow("testg", g)
    cv2.waitKey(0)
    cv2.imshow("testr", r)
    cv2.waitKey(0)
    img_rgb = cv2.merge([r, g, b])
    cv2.imshow("imgbgr", img_rgb)
    cv2.waitKey(0)
    cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB, img_rgb)
    cv2.imshow("imgrgb", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def cv2toGray4():
    # gray = (b + g + r)/3
    # gray = r*0.11 + g*0.59 + b*0.3
    img = cv2.imread("lenna.png", 1)
    # h, w, d = img.shape
    #dst = np.zeros((h, w, 3), np.uint8)
    # for i in range(h):
    #     for j in range(w):
    #         (b, g, r) = img[i, j]
    #         gray = (int(b)+int(g)+int(r))/3
    #         dst[i, j] = np.uint8(gray)

    h, w = img.shape[:2]
    gray = np.zeros([h, w], img.dtype)
    for i in range (h):
        for j in range(w):
            m = img[i, j]
            gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

    cv2.imshow("lenna", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def OTSU_Binary():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("lenna", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Threshold_Binary():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("lenna", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def mean_Threshold_Binary():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, h*w])
    mean = m.sum() / (h*w)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    cv2.imshow("lenna", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Binary():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = np.where(gray >= 127, 1, 0)
    plt.imshow(binary, cmap='gray')
    plt.show()
def compute_Binary():
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    for i in range(rows):
        for j in range(cols):
            if(gray[i, j] <= 127):
                gray[i, j] = 0
            else:
                gray[i, j] = 1
    plt.imshow(gray, cmap = 'gray')
    plt.show()



def nearest_interpolation(img):
    height, width, channels = img.shape
    emptyimage = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh)
            y = int(j/sw)
            emptyimage[i, j] = img[x, y]
    return emptyimage



def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h,src_w = ", src_h, src_w)
    print("dst_h,dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3), dtype = np.uint8)
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5)*scale_x - 0.5
                src_y = (dst_y + 0.5)*scale_y - 0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img



if __name__ == '__main__':
    # cv2toGray1()
    # cv2toGray2()
    # cv2toGray3()
    # cv2toGray4()

    # OTSU_Binary()
    # Threshold_Binary()
    # mean_Threshold_Binary()
    # Binary()
    # compute_Binary()

    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow("bilinear interpolation", dst)
    cv2.waitKey(0)

    # img = cv2.imread("lenna.png")
    # zoom = nearest_interpolation(img)
    # # print(zoom)
    # # print(zoom.shape)
    # cv2.imshow("nearest interpolaton", zoom)
    # cv2.waitKey(0)