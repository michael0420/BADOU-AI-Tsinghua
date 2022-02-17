import cv2
import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt
from 第一周 import billinerInsert

imag = cv2.imread("lenna.png")
# imag = cv2.imread("ll.jpg")
# imag = billinerInsert.bilinearinsert(imag,(400,400))
gray = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
cv2.imshow("origin:",imag)
cv2.imshow("gray:",gray)

# print(gray)
gray_new=gray.copy()
gray_list = gray_new.ravel()
gray_list.sort()
gray_dict = {}
for i in gray_list:
    if i in gray_dict:
        gray_dict[i] += 1
    else:
        gray_dict[i] = 1
count = 0
gray_dict_new = {}
for k,v in gray_dict.items():
    count +=v
    gray_dict_new[k]=count
h,w=gray.shape;
gray_equalization = np.zeros([h, w], gray.dtype)
for hi in range(h):
    for wi in range(w):
        #通过对比接口的直方图,发现接口并未减一
        # gray_equalization[hi,wi]=gray_dict_new[gray[hi,wi]]*256/h/w-1
        gray_equalization[hi,wi]=gray_dict_new[gray[hi,wi]]*256/h/w
cv2.imshow("equalzation_detail:",gray_equalization)
gray_jiekou = cv2.equalizeHist(gray)
cv2.imshow("equalzation_jiekou:",gray_jiekou)
plt.figure()
# plt.subplot(121)
plt.hist(gray_equalization.ravel(),256)
# plt.subplot(122)
plt.hist(gray_jiekou.ravel(),256)
# print(gray_jiekou.ravel())
# print(gray_equalization.ravel())
plt.show()
cv2.waitKey(0)