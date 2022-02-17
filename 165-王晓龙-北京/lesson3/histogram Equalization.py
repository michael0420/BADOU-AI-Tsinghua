from skimage.color import  rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import  cv2
###########################################
# 灰度直方图均匀化
flag =True
#flag =False
if flag:
    img =cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w = img_gray.shape

    #print(h,w)
    # 灰度值方图
    #依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    dst_hist = np.zeros([256],np.uint8)
    dst_img = np.zeros([h, w], img_gray.dtype)
    sum_hist = 0
    # 2 计算灰度直方图的累加直方图
    for i in range(256):
        #print(hist[i])
        sum_hist += float(hist[i]/(h*w))
        value = (sum_hist)*256 - 1
        if value < 0 :
            value = 0
        dst_hist[i] = round(value)


    for i in range(h):
        for j in range(w):
            value  = img_gray[i,j]
            dst_img[i,j] = dst_hist[value]

    dst_hist = cv2.calcHist([dst_img], [0], None, [256], [0, 256])
    #plt.plot(hist)
    plt.plot(dst_hist)
    plt.show()

###########################################
# 灰度直方图均匀化

#flag =True
flag =False
if flag:
    img =cv2.imread("lenna.png")
    # 转换为灰度
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #将灰度图像均衡化
    equal = cv2.equalizeHist(img_gray)
    # 直方图
    hist =cv2.calcHist([equal],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.show()
    #np.hstack() 水平方向合拼
    #条件 图像的大小必须相同
    cv2.imshow("equal", np.hstack([img_gray,equal]))
    cv2.waitKey(0)


###########################################
# 彩色直方图均匀化
#flag =True
#flag =False

if flag:
    img =cv2.imread("lenna.png")
    # 将彩色图像分解通道，然后逐一均匀化
    (b,g,r) =cv2.split(img)
    bEH = cv2.equalizeHist(b)
    gEH = cv2.equalizeHist(g)
    rEH = cv2.equalizeHist(r)

    # 将通道合拼
    res = cv2.merge((bEH,gEH,rEH))

    cv2.imshow("res",res)
    cv2.waitKey(0)