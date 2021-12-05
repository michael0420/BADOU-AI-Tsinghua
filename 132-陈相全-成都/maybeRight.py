#近邻放大，灰度化，二值化
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('lenna.png')
imgg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.subplot(141)
plt.imshow(imgg)#显示原图
print(img.dtype)
w,h,d = img.shape
print(w,h,d)#长，宽，通道
dst = np.zeros([800,800,3],dtype=np.uint8)
w1,h1,d1 = dst.shape
print(w1,h1)
a = w/w1#原图width/800的值
b = h/h1#
print(a,b)

#1：近邻放大
for i in range(w1):#i其实是y轴，因为是行
    for j in range(h1):#j是x
        ii = np.uint(i*a)
        jj = np.uint(j*b)
        #print(ii,jj)
        dst[i,j] = img[ii,jj]
print(dst.dtype)
#print(dst)
dst = cv.cvtColor(dst,cv.COLOR_BGR2RGB)
#dst = dst/255#都变uint8又就不用归一化了

plt.subplot(142)
plt.imshow(dst)
#print(dst[1,1])#[226 137 125]


#二值化
img2 = np.zeros([800,800,3],dtype=np.uint8)
for i in range(w1):
    for j in range(h1):
        if dst[i,j][2]<=128:
            img2[i,j] = [0,0,0]
        else:
            img2[i,j] = [255,255,255]
plt.subplot(143)
plt.imshow(img2)


cv.waitKey(0)

#灰度化
img3 = np.zeros([800,800,3],dtype=np.uint8)
for i in range(w1):
    for j in range(h1):
        img3[i,j] = dst[i,j][1]#我只取RGB中的G通道

plt.subplot(144)
plt.xlabel("灰度化")
plt.imshow(img3)
plt.show()

#双线性插值
src_w,src_h,src_d = img.shape
img4 = np.zeros([800,800,3],dtype=np.uint8)
img4_w,img4_h,img4_d = img4.shape
for dd in range(3):
    for i in range(800):
        for j in range(800):
            #根据0.5那个公式
            src_cx = (j + 0.5)*a-0.5#a,b是什么上面有
            src_cy = (i + 0.5)*b-0.5
            src_x1 = int(src_cx)
            src_y1 = int(src_cy)
            src_x2 = min(src_x1+1,src_w-1)
            src_y2 = min(src_y1+1,src_h-1)

            R1 = (src_x2-src_cx)/(src_x2-src_x1)*img[src_x1,src_y1,dd]+(src_cx-src_x1)/(src_x2-src_x1)*img[src_x2,src_y1,dd]
            R2 = (src_x2 - src_cx) / (src_x2 - src_x1) * img[src_x1, src_y2, dd] + (src_cx - src_x1) / (src_x2 - src_x1) * img[src_x2, src_y2, dd]
            img4[src_cx,src_cy,dd] = (src_y2-src_cy)/(src_y2-src_y1)*R1 +(src_cy-src_y1)/(src_y2-src_y1)*R2

plt.imshow(img4)
plt.show()