



import cv2

def canny(threshold):
    #先高斯滤波，在边缘提取，再进行图像掩模
    img_Gaussian = cv2.GaussianBlur(gray, (3, 3), 0)  #高斯滤波，去除噪声点，使图像平滑
    img_Gaussian = cv2.Canny(img_Gaussian, threshold, threshold)  #边缘检测
    fianl_img = cv2.bitwise_and(img, img, mask=img_Gaussian)  #apertureSize默认值为3
    cv2.imshow("canny_image", fianl_img)


low_threshold = 50
max_threshold = 300

img = cv2.imread("lenna.png",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("canny_image")  #不可少
cv2.createTrackbar("scale", "canny_image", low_threshold, max_threshold, canny)
canny(50)
#cv2.imshow('canny_img', )
cv2.waitKey()


'''

#canny调用版本

img = cv2.imread("lenna.png",1)  #读取图片，彩色读取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #将彩色图片转化为灰度图
img_canny = cv2.Canny(gray, 200, 300)   #Canny处理的图片是灰度图，然后再设置两个阈值
cv2.imshow("img_canny",img_canny)
cv2.waitKey()
'''


