import cv2

img = cv2.imread('photo1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 0) # 高斯滤波
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# edged = cv2.Canny(dilate, 10, 300, 3) # 边缘检测
edged = cv2.Canny(dilate, 30, 120, 3) # 边缘检测

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] # 得到轮廓坐标数组

docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 格局轮廓面积从大到小降序排列
    for c in cnts:
        peri = cv2.arcLength(c, True) # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
    print(docCnt)

for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

cv2.imshow("img", img)
cv2.waitKey(0)

