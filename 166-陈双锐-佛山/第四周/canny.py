import cv2

img = cv2.imread("lenna.png",cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 300)
cv2.imshow("canny", canny)
cv2.waitKey()



