import cv2
import numpy as np
import matplotlib.pyplot as plt

"""调包实现直方图均衡化"""
deal_img = cv2.imread("lenna.png")
gray_img = cv2.cvtColor(deal_img, cv2.COLOR_BGR2GRAY)
dst_img=cv2.equalizeHist(gray_img)
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
# plt.figure()
# plt.plot(hist, "r")  #直方图曲线化
# plt.show()

a=plt.hist(gray_img.ravel(),256,color="r")
b=plt.hist(dst_img.ravel(),256,color="g")
plt.show()

'''
"""直方图均衡化，灰度图详细计算"""
src_img=cv2.imread("../lenna.png")
gray_img=cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)  #转化为灰度图
new_img=np.zeros((gray_img.shape),gray_img.dtype)  #新建图像
h,w=gray_img.shape
sum=0  #定义当前像素数
total_pix=h*w  #计算像素总量
"""遍历每一个像素，并重新赋值（均衡化后的值）"""
for gray_num in range(256):
    """对于每一个像素值，从小到大依次计算总量，根据q=sum/(h*w)*256-1获得当前像素值的均衡化值"""
    for i in range(h):
        for j in range(w):
           if int(gray_img[i,j])==gray_num:
               sum=sum+1
    if q<0:
        q=0
    q=int(sum/total_pix*256-1)  #根据公式q=sum/(h*w)*256-1计算均衡化值
    """对每一个同像素值的像素重新赋值"""
    for x in range(h):
        for y in range(w):
            if int(gray_img[x, y]) == gray_num:
                new_img[x,y]=q
# print(gray_img)
# print(new_img)
# cv2.imshow("new_img",new_img)
# cv2.waitKey(10000)

plt.hist(gray_img.ravel(),256,color="r")
plt.hist(new_img.ravel(),256,color="g")
plt.show()
'''


'''
"""调用库函数，对三通道图像进行直方图均衡化"""
src_img=cv2.imread("lenna.png")
(b,g,r)=cv2.split(src_img)   #分离通道
#分别对三通道进行均衡化
bH=cv2.equalizeHist(b)
gH=cv2.equalizeHist(g)
rH=cv2.equalizeHist(r)
dst_img=cv2.merge((bH,gH,rH))
cv2.imshow("dst_img",dst_img)
# cv2.waitKey(10000)

src_img_show=plt.imread("lenna.png")
plt.subplot(221)
# print("原图显示")
plt.imshow(src_img_show)

dst_img_show=cv2.cvtColor(dst_img,cv2.COLOR_BGR2RGB)
plt.subplot(222)
# print("均衡化显示")
plt.imshow(dst_img_show)
plt.show()
'''

