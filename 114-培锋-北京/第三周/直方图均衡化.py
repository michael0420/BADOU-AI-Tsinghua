import cv2
import numpy
from matplotlib import pyplot as plt

# 1.依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
# 2.计算灰度直方图的累加直方图
# 3.根据累加直方图和直方图均衡化原理得到输入与输出之间的映射关系。
# 4.最后根据映射关系得到结果： dst(x, y) = H'(src(x,y))进行图像变换

gray_level = 256  # 灰度级

#归一化
def pixel_probability(img):

    assert isinstance(img, numpy.ndarray)
    prob = numpy.zeros(shape=(256))

    for rv in img:
        for cv in rv:
            prob[cv] += 1

    r, c = img.shape
    prob = prob / (r * c)
    return prob


#根据归一化结果将原始图像直方图均衡化
def probability_to_histogram(img, prob):

    prob = numpy.cumsum(prob)  # 累计概率
    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

    # 像素值替换
    assert isinstance(img, numpy.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]

    return img


def plot(y, name):
    """
    画直方图，len(y)==gray_level
    :param y: 概率值
    :param name:
    :return:
    """
    plt.figure(num=name)
    plt.bar([i for i in range(gray_level)], y, width=2)

#程序入口
if __name__ == '__main__':

    #读入图片
    img = cv2.imread("F:/cycle_gril/lenna.png",1)
    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #归一化
    prob = pixel_probability(gray)
    plot(prob, "原图直方图")

    #直方图均衡化
    img = probability_to_histogram(gray, prob)
    cv2.imshow("直方图均衡化",img)

    prob = pixel_probability(img)
    plot(prob, "直方图均衡化结果")

    plt.show()
    cv2.waitKey(0)

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
'''
#读入图片
img = cv2.imread("F:/cycle_gril/lenna.png", 1)
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", numpy.hstack([gray, dst]))
cv2.waitKey(0)

'''
'''
# 彩色图像直方图均衡化
img = cv2.imread("F:/cycle_gril/lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''

