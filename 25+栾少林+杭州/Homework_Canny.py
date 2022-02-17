import cv2

src_image=cv2.imread("lenna.png")
gray_image=cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY)
dst_image1=cv2.Canny(gray_image,200,300)
# dst_image2=cv2.Canny(gray_image,100,300)
# dst_image3=cv2.Canny(gray_image,200,200)
cv2.imshow("Canny",dst_image1)
cv2.waitKey(10000)

