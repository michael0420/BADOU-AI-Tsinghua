import cv2 as cv
import numpy as np

# img = cv.imread('lenna.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = np.array(
[
    [1, 3, 9, 9, 8],
    [2, 1, 3, 7, 3],
    [3, 6, 0, 6, 4],
    [2, 9, 2, 6, 0]
]
)
h, w = img.shape


def all_np(img, h, w):
    # 获取图片展开array
    arr = img.flatten()
    # 获取图片像素属性
    keys = np.unique(arr)
    # 像素属性排序
    keys = np.sort(keys)
    result = {}
    temp = 0
    for k in keys:
        # 计算对应像素属性出现的次数
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        # 计算像素属性Pi
        pi = np.float(v / (h * w))
        # 计算像素属性累加 Pi
        sumPi = np.float(temp + pi)
        temp = sumPi
        # 构建返回字典
        result[k] = int(sumPi * 256 - 1)
    return result


# 进行图片均衡化处理
def convert_hist(img, h, w, result):
    for i in range(h):
        for j in range(w):
            key = img[i, j]
            img[i, j] = result[key]

    return img


result = all_np(img, h, w)
new_image = convert_hist(img, h, w, result)

# image2 = cv.equalizeHist(img)
# cv.imshow("new image", new_image)
# cv.imshow("new image2", image2)
# cv.waitKey(0)
# todo need confirm whether this is make by precision
# my result
# [[ 50 139 255 255 216]
#  [ 88  50 139 203 139]
#  [139 191  24 191 152]
# [ 88 255  88 191  24]]

# expectation is
[
    [50,132,255,255,224],
    [91,50,132,204,132],
    [132,194,30,194,142],
    [92,255,91,194,30]
]

print(new_image)


