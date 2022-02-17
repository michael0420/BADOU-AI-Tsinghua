import cv2
import numpy as np

def getdingdian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt


img = cv2.imread('photo1.jpg')
result3 = img.copy()
src = getdingdian(img).reshape((4,2))
src=np.float32([src[0],src[3],src[1],src[2]])
# minx=src[np.argmin(src[:,0])][0]
# maxx=src[np.argmax(src[:,0])][0]
# miny=src[np.argmin(src[:,1])][1]
# maxy=src[np.argmax(src[:,1])][1]
# hight=maxy-miny
# width=maxx-minx
# dst = np.float32([[0, 0], [hight, 0], [0,width], [hight, width]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 生成透视变换矩阵；进行透视变换
print(src)
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (377, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)




