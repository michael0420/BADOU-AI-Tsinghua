# encoding: utf-8

import cv2
import numpy as np

'''链接相同特征点'''


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    '''把图片并列展示出来'''
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


'''首先导入一组照片，首先查找和描述关键点'''
img1 = cv2.imread("cs01.png")
img2 = cv2.imread("cs02.png")
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)  #
matches = bf.match(des1,des2)   #Match(des1, des2)  #

# goodMatch = []
# for m, n in matches:
#     if m.distance < 0.50 * n.distance:
#         goodMatch.append(m)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

cv2.imshow('img3',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
