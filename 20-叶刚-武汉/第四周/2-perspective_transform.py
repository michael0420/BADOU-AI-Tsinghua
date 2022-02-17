"""
@GiffordY
使用透视变换算法，实现图像矫正
"""

import cv2 as cv
import numpy as np


# 1、读入图像
img_src = cv.imread('photo1.jpg', 1)

# 2、灰度化、高斯滤波、灰度膨胀
img_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_dilate = cv.dilate(img_blur, kernel)

# 3、边缘检测、轮廓检测
img_edges = cv.Canny(img_dilate, 50, 150, 3)
cv.imshow('edges', img_edges)
contours, hierarchy = cv.findContours(img_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 显示轮廓
img_src2 = img_src.copy()
if len(contours) > 0:
    for i in range(len(contours)):
        cv.drawContours(img_src2, contours, i, (0, 0, 255), 1, 8)

# 4、轮廓筛选
detected_points = None
if len(contours) > 0:
    # 根据轮廓面积从大到小排序
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for contour in contours:
        # 计算轮廓周长
        # arc_len = cv.arcLength(contour, closed=True)
        # 计算轮廓面积
        area = cv.contourArea(contour)
        # 轮廓多边形拟合，得到轮廓的角点
        points = cv.approxPolyDP(contour, 5, closed=True)
        # 使用面积和角点筛选合适的轮廓
        if area > 1000 and len(points) == 4:
            detected_points = points
            break
print('原始三维角点坐标：', detected_points)
detected_points = detected_points[:, 0, :]
print('原始二维角点坐标：', detected_points)


# 绘制筛选出的角点
for point in detected_points:
    cv.circle(img_src2, point, 10, (0, 255, 0), 1, 8, 0)
# 显示
cv.imshow('detection result', img_src2)

# 5、计算目标点的坐标
rect = cv.minAreaRect(detected_points)
print(rect)
(cx, cy) = rect[0]
(w, h) = rect[1]
angle = rect[2]

dst_point1 = [cx-w/2, cy-h/2]
dst_point2 = [cx+w/2, cy-h/2]
dst_point3 = [cx-w/2, cy+h/2]
dst_point4 = [cx+w/2, cy+h/2]
dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
print("目标图4个角点dst_points = ", dst_points)

# 对原始角点的坐标进行整理，便于与目标点对应上
tmp_points = {}
for point in detected_points:
    if point[1] < cy:
        if point[0] < cx:
            tmp_points[1] = point
        else:
            tmp_points[2] = point
    else:
        if point[0] < cx:
            tmp_points[3] = point
        else:
            tmp_points[4] = point

points_list = list(tmp_points.items())
points_list.sort(reverse=False)
src_points = np.float32([point[1] for point in points_list])
print("原图4个角点src_points = ", src_points)

# 6、生成透视变换矩阵；进行透视变换
m = cv.getPerspectiveTransform(src_points, dst_points)
print('透视变换矩阵m = ', m)
src_height = img_src.shape[0]
src_width = img_src.shape[1]
img_dst = cv.warpPerspective(img_src, m, dsize=(src_width, src_height), flags=cv.INTER_LINEAR)

# 7、显示结果
cv.imshow('img_src', img_src)
cv.imshow('img_dst', img_dst)
cv.waitKey(0)
cv.destroyAllWindows()

