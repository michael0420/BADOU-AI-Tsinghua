#双线性插值法

import numpy as np
import cv2

def bilinear(img):
    img_h, img_w, img_c = img.shape
    tar = np.zeros([1000, 1000, img_c], dtype=np.uint8)
    tar_h, tar_w = tar.shape[:2]
    print(tar_h,tar_w,img_c)
    prop_x, prop_y = float(img_w) / tar_w, float(img_h) / tar_h
    print(prop_y, prop_x)
    for c in range(img_c):
        for tar_y in range(tar_h):
            for tar_x in range(tar_w):
                #中心对称
                src_x = (tar_x + 0.5) * prop_x - 0.5
                src_y = (tar_y + 0.5) * prop_y - 0.5

                # 找到四个像素顶点 x1, x2, y1, y2
                x1 = int(np.floor(src_x))
                x2 = min(x1 + 1, img_w - 1)
                y1 = int(np.floor(src_y))
                y2 = min(y1 + 1, img_h - 1)

                # y = (x2 - x) * y1 + (x - x1) * y2    相邻两个像素点之间差为1，故分母为0
                f_1 = (x2 - src_x) * img[y1, x1, c] + (src_x - x1) * img[y1, x2, c]  # f(R1)
                f_2 = (x2 - src_x) * img[y2, x1, c] + (src_x - x1) * img[y2, x2, c]  # f(R2)
                #y = (y2 - y) * y1 + (y - y1) * y2
                tar[tar_y, tar_x, c] = int((y2 - src_y) * f_1 + (src_y - y1) * f_2)
    return  tar

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    tar = bilinear(img)
    cv2.imshow("tar",tar)
    cv2.waitKey()



