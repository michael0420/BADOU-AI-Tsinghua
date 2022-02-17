# author:
# createTime: 2021/11/24 0:24
# describe:bilinear interpolation
import math

import cv2
import numpy as np

"""
           y
    ------------->
    |       |
    |---p1-----p2---   tmp1
 x  |-------|-------   target
    |---p4--|--p3---   tmp2
    V
"""
if __name__ == '__main__':
    img = cv2.imread("pic/lenna.png")
    h, w = img.shape[:2]
    print(h,w)
    # target_size = (300, 300)
    target_size = (700, 700)
    th, tw = target_size[0], target_size[1]
    # th, tw = h, w
    target_img = np.zeros((th, tw, 3), np.uint64)
    for c in range(3):
        for x in range(th):
            for y in range(tw):
                corr_x = (x + 0.5) * float(h / th) - 0.5
                corr_x = 0 if corr_x < 0 else corr_x
                corr_y = (y + 0.5) * float(w / tw) - 0.5
                corr_y = 0 if corr_y < 0 else corr_y
                p1 = (math.floor(corr_x), math.floor(corr_y))  # 顺时针排序
                p2 = (p1[0], min(p1[1] + 1, w-1))
                p3 = (min(p1[0] + 1, h-1), min(p1[1] + 1, w-1))
                p4 = (min(p1[0] + 1, h-1), p1[1])
                # y 方向
                # [ f(tmp1)-f(p1) ] / [f(p2)-f(p1) ] = tmp1[1] - p1[1] / [ p2[1]-p1[1] ]
                # f(tmp1) = [tmp1[1] - p1[1] ]/ [ p2[1]-p1[1] ] *[f(p2)-f(p1) ] + f(p1)
                # [ f(tmp2)-f(p4) ] / [f(p3)-f(p4) ] = tmp2[1] - p4[1] / [ p3[1]-p4[1] ]
                # f(tmp2)= tmp2[1] - p4[1] / [ p3[1]-p4[1] ]*[f(p3)-f(p4) ] + f(p4)
                # f(p1)=img[p1[0],p1[1]];f(p2)=img[p2[0],p2[1]];f(p3)=img[p3[0],p3[1]];tmp1[1]=y;tmp1[0]=p1[0];tmp2[0]=p4[0];tmp2[1]=y;
                # x方向
                # [ f(tmp2) - f(tmp1) ] / [f(target)- f(tmp1) ] =[ tmp2[0] - tmp1[0] ]/ [ target[0] - tmp1[0] ]
                # f(target) =[ f(tmp2) - f(tmp1) ]/ [ tmp2[0] - tmp1[0] ] * [ target[0] - tmp1[0] ]+f(tmp1)
                try:
                    f_tmp1 = (corr_y - p1[1]) * (int(img[p2[0], p2[1], c]) - int(img[p1[0], p1[1], c])) / (p2[1] - p1[1]+0.00001) + int(img[p1[0], p1[1], c])
                    f_tmp2 = (corr_y - p4[1]) * (int(img[p3[0], p3[1], c]) - int(img[p4[0], p4[1], c])) / (p3[1] - p4[1]+0.00001) + int(img[p4[0], p4[1], c])
                    target_img[x, y, c] = f_tmp1 + (f_tmp2 - f_tmp1) / (p4[0] - p1[0]+0.00001) * (corr_x - p1[0])
                except Exception as e:
                    print(e)
                    print(corr_x,corr_y,p1,p2,p3,p4)
                    break

    target_img = target_img.astype(np.uint8)
    cv2.imshow("target_img", target_img)
    cv2.imshow("src_img", img)
    cv2.waitKey()
