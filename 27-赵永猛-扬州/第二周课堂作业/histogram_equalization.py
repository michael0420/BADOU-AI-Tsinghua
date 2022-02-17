import numpy as np
import cv2
import collections


def gray_hist(gray_img):
    # gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    w, h =np.shape(gray_img)
    print(type(gray_img))
    # cv2.imshow("gray image", gray_img)
    # w, h = np.shape(np.array(gray_img))
    # A = [] # 统计每个像素值对应的像素点数量

    B = dict(collections.Counter(gray_img.flatten())) # 统计每个像素值及对应的像素点数；flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
    print('每个灰度值及对应的像素点个数：\n', type(B))
    # print('每个灰度值及对应的像素点个数：\n', B[28])
    # D = sorted(B, key=B.__getitem__, reverse=False)
    D = [key for (key, value) in sorted(B.items(), reverse=False)] # 键排序
    print('键排序D：\n', D)
    E = [value for (key, value) in sorted(B.items(), reverse=False)] # 根据键排序得到的对应的值
    print('根据键排序得到的对应的值E：\n', E)
    new_list = [x / (w*h) for x in E] # Pi=Ni/image
    print('Pi', new_list)
    sum = []  # Ni
    sump = 0
    length_key = len(D)
    for i in range(length_key):
        sum.append(i)
        sum[i] = sump + new_list[i]
        sump = sum[i]
    sum_new_list = [int(x * 256-1) for x in sum]
    print('四舍五入：\n', sum_new_list)
    test = np.zeros((w, h), dtype=np.uint8) # 创建w*h的0值矩阵
    for j in range(w):
        for k in range(h):
            for l in range(length_key):
                if gray_img[j, k] == D[l]:
                    test[j, k] = sum_new_list[l] # 将像素点替换为对应转化后的像素值

    return test


if __name__ == '__main__':
    image = cv2.imread('lenna.png')
    (b, g, r) = cv2.split(image)
    bH = gray_hist(b)
    gH = gray_hist(g)
    rH = gray_hist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", result)
    # gray_hist(image)
    cv2.waitKey(0)
