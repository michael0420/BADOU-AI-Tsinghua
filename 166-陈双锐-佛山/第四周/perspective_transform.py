import cv2
import numpy as np

src_img = cv2.imread("photo1.jpg", cv2.IMREAD_COLOR)
src_cor = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst_cor = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

matrix = cv2.getPerspectiveTransform(src_cor,dst_cor)
print("warp matrix:", matrix)

dst_img = cv2.warpPerspective(src_img, matrix,(337,488))
cv2.imshow("src_img",src_img)
cv2.imshow("dst_img",dst_img)
cv2.waitKey()










