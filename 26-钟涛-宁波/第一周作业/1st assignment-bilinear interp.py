"""
双线性插值法
"""
import cv2
import numpy as np

"""
函数功能:
双线性插值法实现图像缩放
参数：
src_image--原图像
dst_w--目标图像宽度
dst_h--目标图像高度
"""
def bilinear_interp(src_image,dst_h,dst_w):
    src_h,src_w,chanels = np.shape(src_image);
    dst_image = np.zeros((dst_h,dst_w,chanels),np.uint8);
    for c in range(chanels):
        for i in range(dst_h):
            for j in range(dst_w):
                h = (i+0.5) * float((src_h/dst_h)) - 0.5;
                w = (j+0.5) * float((src_w/dst_w)) - 0.5;
                h1 = int(h);
                h2 = min((h1 + 1),(src_h - 1));
                w1 = int(w);
                w2 = min((w1 + 1),(src_w - 1));
                temp1 = (w2 - w) * src_image[h1,w1,c] + (w - w1) * src_image[h1,w2,c];
                temp2 = (w2 - w) * src_image[h2,w1,c] + (w - w1) * src_image[h2,w2,c];
                dst_image[i,j,c] = int((h2 - h) * temp1 + (h -h1) * temp2);
    return dst_image

test_image = cv2.imread('lenna.png');
print(test_image.shape);
image1 = bilinear_interp(test_image,200,200);
print(image1.shape);
image2 = bilinear_interp(test_image,800,800);
print(image2.shape);
cv2.imshow("bilinear interp1",image1);
cv2.imshow("bilinear interp2",image2);
cv2.imshow("source image",test_image);
cv2.waitKey(0);