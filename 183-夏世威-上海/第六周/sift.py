import cv2
import numpy as np

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# 特征点检测
keypoints, descriptor = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, keypoints, img)
cv2.imshow("keypoints", img)


# ****************************************************
# ****************************************************
def draw(img1, kp1, img2, kp2, matchs):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img[:h1, :w1] = img1
    img[:h2, w1:w1 + w2] = img2
    p1 = [kpp.queryIdx for kpp in matchs]
    p2 = [kpp.trainIdx for kpp in matchs]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))
    return img


# 特征点匹配
img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
good_match = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_match.append(m)

img = draw(img1, kp1, img2, kp2, good_match[:20])
cv2.imshow("match", img)
cv2.waitKey(0)
