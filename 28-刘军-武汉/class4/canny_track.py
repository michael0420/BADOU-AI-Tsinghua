import cv2
def CannyThreshold(lowThreshold):
    detected_edges_img = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges_img = cv2.Canny(detected_edges_img,lowThreshold,
                                   lowThreshold*ratio,apertureSize=kernel_sizes)
    dst = cv2.bitwise_and(img,img,mask=detected_edges_img)
    cv2.imshow('Canny',dst)

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
min_Threshold = 0
max_lowThreshold = 255
kernel_sizes = 3
ratio = 3
cv2.namedWindow('canny demo')
cv2.createTrackbar('Min threshold','canny demo',min_Threshold,max_lowThreshold,CannyThreshold)
CannyThreshold(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
