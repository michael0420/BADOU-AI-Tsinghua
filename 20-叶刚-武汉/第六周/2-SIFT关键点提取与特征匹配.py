"""
@GiffordY
SIFT特征点（关键点）提取和特征匹配
"""

import numpy as np
import cv2

"""
OpenCV SIFT关键点提取并显示，步骤：
第一步：读入图片
第二步：进行灰度化（非必要）
第三步：使用cv2.SIFT_create()实例化sift函数
    （注：OpenCV 4.X版本使用cv2.SIFT_create()函数，OpenCV 2.X版本使用cv2.xfeatures2d.SIFT_create()函数）
第四步：使用sift.detectAndCompute()函数检测关键点并计算描述符 （sift.detect()函数仅检测关键点）
第五步：使用cv2.drawKeypoints()函数进行关键点绘图
第六步：在原图上显示关键点
"""
# 1、
img = cv2.imread('../00-data/images/iphone1.png', 1)
# 2、
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 3、
sift = cv2.SIFT_create()
# 4、
keypoints, descriptors = sift.detectAndCompute(img, mask=None)
# 5、
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向
img = cv2.drawKeypoints(img, keypoints, outImage=img, color=(-1, -1, -1),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 6、
print('len(keypoints) = ', len(keypoints))
print('descriptors.shape = ', descriptors.shape)
cv2.imshow('keypoints', img)

"""
OpenCV SIFT特征点提取与特征匹配
"""
# 1、读入模板图像和观测图像
img1 = cv2.imread('../00-data/images/iphone1.png', 1)

img2 = cv2.imread('../00-data/images/iphone2.png', 1)

# 2、分别提取关键点和计算描述符
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, mask=None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, mask=None)

# 3、特征匹配
# 创建暴力匹配器对象，这个匹配器使用L2范数（欧式距离）度量匹配度
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_L2)
# 从查询集中为每个描述符查找k个最佳匹配项
# matches = bf.knnMatch(descriptors1, descriptors2, k=1)
matches = bf.match(descriptors1, descriptors2)
print('len(matches) = ', len(matches))

# 4、对匹配的结果按照距离进行排序操作
matches = sorted(matches, key=lambda x: x.distance)

# 5、绘制匹配结果
# 可以不用自己拼接两幅图像，使用cv2.drawMatches()函数会自己创建拼接图像
# h1, w1 = img1.shape[:2]
# h2, w2 = img2.shape[:2]
# img_result = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
# img_result[:h1, :w1] = img1
# img_result[:h2, w1:w1 + w2] = img2

# 取匹配结果最好的前10对特征点，绘制图像
img_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None,
                             matchColor=(-1, -1, -1), flags=2)

cv2.imshow('img_result', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
