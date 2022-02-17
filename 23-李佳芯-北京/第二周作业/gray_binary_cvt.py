import cv2
import numpy as np

"""
# 灰度化（调接口）
def gray_cvt(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 灰度化
    return gray_image
"""


# 灰度化(具体实现)
def gray_cvt(image):
    h = image.shape[0]  # 获取图片的高
    w = image.shape[1]  # 获取图片的宽
    gray_image = np.zeros((h, w), dtype=image.dtype)
    for i in range(h):
        for j in range(w):
            gray_image[i, j] = int(image[i, j][2] * 0.3 + image[i, j][1] * 0.59 + image[i, j][0] * 0.11)
    return gray_image


"""
# 二值化（调接口）
def binary_cvt():
    gray_image = gray_cvt(image)
    ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)  # 二值化
    return binary_image
"""


# 二值化(具体实现)
def binary_cvt():
    gray_image = gray_cvt(image)  # 获取灰度图
    x = gray_image.shape[0]  # 获取图片的高
    y = gray_image.shape[1]  # 获取图片的宽
    binary_image = np.zeros((x, y), dtype=gray_image.dtype)
    for i in range(x):
        for j in range(y):
            if gray_image[i, j] <= 127:
                binary_image[i, j] = 0
            else:
                binary_image[i, j] = 255
    return binary_image


if __name__ == '__main__':
    image = cv2.imread("lenna.png")  # 读取图像
    gray_cvt_image = gray_cvt(image)  # 灰度化
    cv2.imshow("gray_cvt_image", gray_cvt_image)  # 显示灰度化后的图像
    cv2.imwrite("gray_cvt_image.png", gray_cvt_image)  # 保存灰度化后的图像
    binary_cvt_image = binary_cvt()  # 二值化
    cv2.imshow("binary_cvt_image", binary_cvt_image)  # 显示二值化后的图像
    cv2.imwrite("binary_cvt_image.png", binary_cvt_image)  # 保存二值化后的图像
    cv2.waitKey()  # 等待操作
    cv2.destroyAllWindows()  # 关闭显示图像的窗口
