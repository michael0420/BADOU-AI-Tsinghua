"""
最邻近差值
"""
import cv2
import numpy as np

"""
函数功能:
最近邻插值法实现图像缩放
参数：
src_image--原图像
dst_w--目标图像宽度
dst_h--目标图像高度
"""
def function(src_image,dst_h,dst_w):
    src_h,src_w,chanels = np.shape(src_image);
    dst_image = np.zeros((dst_h,dst_w,chanels),np.uint8);
    for i in range(dst_h):
        for j in range(dst_w):
            h = int(i *(src_h/dst_h) );
            w = int(j * (src_w/dst_w));
            dst_image[i,j] = src_image[h,w];
    return dst_image

test_image = cv2.imread('lenna.png');
image1 = function(test_image,200,200);
print(image1.shape);
image2 = function(test_image,800,800);
print(image2.shape);
cv2.imshow("nearest interp1",image1);
cv2.imshow("nearest interp2",image2);
cv2.imshow("source image",test_image);
cv2.waitKey(0);
