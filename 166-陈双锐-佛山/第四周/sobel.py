import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png",cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_sobel_x = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
img_sobel_y = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
img_laplace = cv2.Laplacian(gray,cv2.CV_64F,ksize=3)
img_canny = cv2.Canny(gray,100,300)
plt.subplot(221),plt.imshow(img_sobel_x, "gray"),plt.title("img_sobel_x")
plt.subplot(222),plt.imshow(img_sobel_y, "gray"),plt.title("img_sobel_y")
plt.subplot(223),plt.imshow(img_laplace, "gray"),plt.title("img_laplace")
plt.subplot(224),plt.imshow(img_canny, "gray"),plt.title("img_canny")
plt.show()
