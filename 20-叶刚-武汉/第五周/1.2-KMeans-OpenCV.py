"""
@GiffordY
应用OpenCV框架提供的KMeans接口，对彩色图像或灰度图像的像素进行聚类（图像分割）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
函数参数：
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS；和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心坐标的输出矩阵，每个集群中心为一行数据
返回值：
    retval：即compactness紧密度，返回每个点到相应重心的距离的平方和
    bestLabels：表示每个样本的聚类标签索引，如0、1、2...K-1
    centers：表示集群中心坐标的输出矩阵，每个集群中心为一行数据
'''


def imageClustering(src_img, K, bestLabels, criteria, attempts, flags, centers=None, resultType='LABELS'):
    """
    函数功能：应用OpenCV框架提供的KMeans接口，对彩色图像或灰度图像的像素进行聚类（图像分割）
    输入参数：
        src_img：输入图像，彩色图像或者灰度图像
        resultType：输出的聚类结果图像的类型，LABELS表示输出图像的像素值为类别标签；CENTERS表示输出图像的像素值为聚类中心值
        其他参数：同OpenCV kmeans()函数
    返回值：
        聚类结果图像，unit8型
    """
    # 检查resultType参数
    resultTypes = ['LABELS', 'CENTERS']
    if resultType not in resultTypes:
        raise ValueError("resultType must be 'LABELS' or 'CENTERS', " f"got '{resultType}' instead.")

    # 检查图像是否为空
    max_val = np.max(src_img)
    if max_val is None:
        raise ValueError('Error, input img is None!')

    # 2.将图像的每个像素作为一个样本点，并将像素值转换为float32类型
    # 使用reshape函数展开，灰度图像展开为（n_samples,1）,彩色图像展开为（n_samples,3）
    # n_samples = rows * cols，填-1时会自动计算；order='C'，表示按行展开
    ret = src_img.shape
    if len(ret) == 3:
        data = np.reshape(src_img, (-1, 3), order='C')
    else:
        data = np.reshape(src_img, (-1, 1), order='C')
    data = np.float32(data)

    # 3.使用OpenCV的Kmeans()函数对每个像素进行聚类
    compactness, pred_labels, pred_centroids = cv2.kmeans(data, K, None, criteria, attempts, flags)

    # 4.生成聚类后的图像，并将图像返回uint8类型
    # 如果resultType为'LABELS'，返回单通道图像，且每个像素的像素值为所在簇的标签
    if resultType == 'LABELS':
        pred_labels = pred_labels.flatten()
        dst_img = pred_labels.reshape((src_img.shape[0], src_img.shape[1]))
        dst_img = np.uint8(dst_img)
    # 否则，返回灰度图返回单通道图像，彩色图返回三通道图像，且每个像素的像素值为所在簇的质心坐标值
    else:
        pred_centroids = np.uint8(pred_centroids)
        tmp = pred_centroids[pred_labels.flatten()]
        dst_img = np.reshape(tmp, src_img.shape)
    return dst_img


if __name__ == '__main__':
    # 设置KMeans函数的参数
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS
    resultTypes = ['LABELS', 'CENTERS']

    # 读入灰度图像
    img_gray = cv2.imread('../00-data/images/lenna.png', 0)
    img_gray_dst = imageClustering(img_gray, K, None, criteria, attempts, flags,
                                   centers=None, resultType=resultTypes[0])

    # 读入彩色图像
    img_color = cv2.imread('../00-data/images/lenna.png', 1)
    img_color_dst = imageClustering(img_color, K, None, criteria, attempts, flags,
                                    centers=None, resultType=resultTypes[1])

    # 显示图像
    print('img_gray.shape = {}, img_gray_dst.shape = {}'.format(img_gray.shape, img_gray_dst.shape))
    print('img_color.shape = {}, img_color_dst.shape = {}'.format(img_color.shape, img_color_dst.shape))

    titles = [u'原始灰度图像', u'灰度聚类图像', u'原始彩色图像', u'彩色聚类图像']
    # 设置字体为SimHei可以正常显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']

    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    if len(img_color_dst.shape) == 3:
        img_color_dst = cv2.cvtColor(img_color_dst, cv2.COLOR_BGR2RGB)

    images = [img_gray, img_gray_dst, img_color, img_color_dst]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
