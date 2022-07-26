import cv2

low_threshold = 0
max_low_threshold = 100
ratio = 2
kernel_size = 3

cv2.namedWindow('Canny')
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def call_back(threshold):
    # 高斯模糊降噪
    image = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny 边缘检测
    image = cv2.Canny(image, threshold, threshold * ratio, apertureSize=kernel_size)
    # 填充原始像素
    image = cv2.bitwise_and(img, img, mask=image)
    cv2.imshow("Canny", image)


cv2.createTrackbar('threshold', 'Canny', low_threshold, max_low_threshold, call_back)

call_back(low_threshold)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
