#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/16 21:27:46
@Author : Luposs_idxsglin

Canny 边缘检测算法的具体实现
'''


import os
import numpy as np
import cv2


class Canny:
    """Canny 边缘检测, 对象通常为灰度图
    
    Step 1. 使用高斯滤波去除输入图像中的噪声，高斯模糊核，指定x,y轴半径大小，按半径和高斯分布函数计算像素值，最后核内总像素点权重归一化
    Step 2. 计算高斯滤波后的图像的x\y方向的梯度，可用sobel核或其他计算
    Step 3. 考虑梯度方向邻近点，邻近的window的大小设置，插值临近点的方法等都可以自由调整，抑制局部非最大梯度点
    Step 4. 双阈值法，大于高阈值的认为一定是边缘，小于低阈值的认为一定不是，处于中间的像素如果是在高阈值边缘点联通，则也认为是边缘点。
    """
    def __init__(self, img: np.ndarray, sigma: float, ksize: int=None, low_thres=None, high_thres=None) -> None:
        self.img = img
        self.kernel_blur = self._gaussian_blur(sigma, ksize)
        self.img_blured = self._conv(img, self.kernel_blur)
        self.img_grad, self.img_supress = self._nonMax_supression(self.img_blured)
        self.detecd_img = self.two_thres_supress(low_thres, high_thres)

    def _gaussian_blur(self, sigma: float, ksize: int=None):
        """生成高斯滤波器
        Parameter
        ---------
        sigma : 高斯方差
        """
        # 用高斯分布函数生成高斯模糊核
        if ksize is None:
            dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
            if dim % 2 == 0:  # 最好是奇数,不是的话加一
                dim += 1
        else:
            dim = ksize
        Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
        # 单位半径序列
        tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
        n1 = 1/(2*np.pi*sigma**2)  # 计算高斯核
        n2 = -1/(2*sigma**2)
        for i in range(dim):
            for j in range(dim):
                Gaussian_filter[i, j] = n1*np.exp(n2*(tmp[i]**2+tmp[j]**2))
        # 所有点权重归一化
        Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
        return Gaussian_filter

    def _conv(self, input, kernel, padding="SAME"):
        """对图像进行滤波
        Parameters
        ----------
        input : np.ndarray
            灰度图
        kernel : np.ndarray
            滤波器, shape=(ksize, ksize), ksize 必为奇数
        padding : str
            choices = ['SAME', 'VALID']

        """
        H, W = input.shape
        ksize = kernel.shape[0]
        assert (ksize- 1) % 2 == 0, "ksize must be odd number!"
        # 根据填充模式决定输出大小
        if padding == "SAME":
            output_size = (H, W)
            # 对input做zero-padding
            input_pad = np.zeros((H + ksize - 1, W + ksize - 1))
            origin_idx = int(np.ceil((ksize - 1) / 2))
            input_pad[origin_idx:-origin_idx, origin_idx:-origin_idx] = input
            input = input_pad
        else:
            output_size = (H - ksize + 1, W - ksize + 1)

        conv_ls = []
        sta_idx = int(np.ceil((ksize - 1) / 2))
        # 从左到右，从上到下
        for i in range(sta_idx + output_size[0]-1):
            for j in range(sta_idx + output_size[1]-1):
                conv_ls.append((kernel * input[i:i+ksize, j:j+ksize]).sum())

        return np.array(conv_ls).reshape(output_size[0], output_size[1])     

    def _nonMax_supression(self, img_blur):
        """对图像求梯度，在梯度方向上进行非极大值抑制
        Parameter
        ---------
        img_blur : np.ndarray
            经过高斯模糊的灰度图
        """
        # 1. calculate gradient
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        img_grad_x = img_grad_y = np.zeros_like(img_blur)
        img_pad = np.pad(img_blur, ((1, 1), (1, 1)), 'constant')
        for i in range(img_blur.shape[0]):
            for j in range(img_blur.shape[1]):
                img_grad_x[i, j] = (img_pad[i:i+3, j:j+3] * sobel_x).sum()
                img_grad_y[i, j] = (img_pad[i:i+3, j:j+3] * sobel_y).sum()
        img_grad = np.sqrt(img_grad_x ** 2 + img_grad_y ** 2)
        # 数值稳定性
        img_grad_x[img_grad_x == 0] = 1e-7
        tanh_angle = img_grad_y / img_grad_x

        # 2. 非极大值抑制
        img_supress = np.zeros_like(img_grad)
        """一维线性插值计算领域点，根据角度tan \\theta 判断，与八条边相交，
        梯度方向两两对称，则共4种不重复情况"""
        for i in range(1, img_grad.shape[0]-1):
            for j in range(1, img_grad.shape[1] - 1):
                # 当后续判断该点为非极大值点时更改为False
                flag = True
                # 要进行对比的搜索领域
                search_radius = img_grad[i-1:i+2, j-1:j+2]
                if tanh_angle[i, j] <= -1:
                    # `\_
                    num1 = search_radius[0, 1] + (search_radius[0, 1] - search_radius[0, 0]) / tanh_angle[i, j]
                    num2 = search_radius[2, 1] + (search_radius[2, 1] - search_radius[2, 2]) / tanh_angle[i, j]
                    if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                        flag = False
                elif tanh_angle[i, j] >= 1:
                    # _/`
                    num1 = (search_radius[0, 2] - search_radius[0, 1]) / tanh_angle[i, j] + search_radius[0, 1]
                    num2 = search_radius[2, 1] + (search_radius[2, 0] - search_radius[2, 1]) / tanh_angle[i, j]
                    if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                        flag = False
                elif tanh_angle[i, j] > 0:
                    # 左下，右上边
                    num1 = search_radius[1, 2] + (search_radius[0, 2] - search_radius[1, 2]) * tanh_angle[i, j]
                    num2 = search_radius[1, 0] + (search_radius[2, 0] - search_radius[1, 0]) * tanh_angle[i, j]
                    if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                        flag = False
                elif tanh_angle[i, j] < 0:
                    # 左上，右下边
                    num1 = search_radius[1, 0] + (search_radius[1, 0] - search_radius[0, 0]) * tanh_angle[i, j]
                    num2 = search_radius[1, 2] + (search_radius[1, 2] - search_radius[2, 2]) * tanh_angle[i, j]
                    if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                        flag = False
                if flag:
                    # 存储极大值
                    img_supress[i, j] = img_grad[i, j]
        return img_grad, img_supress

    def two_thres_supress(self, low_thres=None, high_thres=None):
        """双阈值
        Parameters
        ----------
        low_thres : int
            低阈值
        high_thres : int
            高阈值
        """
        if low_thres is None:
            low_thres = self.img_grad.mean() * 0.5
        if high_thres is None:
            high_thres = low_thres * 3
        poss_edges_idx = np.argwhere(self.img_supress >= high_thres)
        poss_idxs = np.array([poss_edges_idx[0], poss_edges_idx[1]]).reshape(-1, 2).tolist()
        # 外圈没有剔除
        img_supress = np.where(self.img_supress >=high_thres, 255, np.where(self.img_supress <= low_thres, 0, self.img_supress))
        # 外圈赋予原值
        img_supress[[0,-1], :] = self.img_supress[[0, -1], :]
        img_supress[:, [0,-1]] = self.img_supress[:, [0, -1]]

        while not len(poss_idxs) == 0:
            temp_1, temp_2 = poss_idxs.pop()  # 出栈
            a = img_supress[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
            if (a[0, 0] < high_thres) and (a[0, 0] > low_thres):
                img_supress[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
                poss_idxs.append([temp_1-1, temp_2-1])  # 进栈
            if (a[0, 1] < high_thres) and (a[0, 1] > low_thres):
                img_supress[temp_1 - 1, temp_2] = 255
                poss_idxs.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_thres) and (a[0, 2] > low_thres):
                img_supress[temp_1 - 1, temp_2 + 1] = 255
                poss_idxs.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_thres) and (a[1, 0] > low_thres):
                img_supress[temp_1, temp_2 - 1] = 255
                poss_idxs.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_thres) and (a[1, 2] > low_thres):
                img_supress[temp_1, temp_2 + 1] = 255
                poss_idxs.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_thres) and (a[2, 0] > low_thres):
                img_supress[temp_1 + 1, temp_2 - 1] = 255
                poss_idxs.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_thres) and (a[2, 1] > low_thres):
                img_supress[temp_1 + 1, temp_2] = 255
                poss_idxs.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_thres) and (a[2, 2] > low_thres):
                img_supress[temp_1 + 1, temp_2 + 1] = 255
                poss_idxs.append([temp_1 + 1, temp_2 + 1])
    
        for i in range(img_supress.shape[0]):
            for j in range(img_supress.shape[1]):
                if img_supress[i, j] != 0 and img_supress[i, j] != 255:
                    img_supress[i, j] = 0
        return img_supress


def canny_api(img, thres1, thres2, gksize=3):
    """调用opencv接口
    Parameters
    -----------
    img : np.ndarray
        gray image
    thres1 : int
        low threshold
    thres2 : int
        high threshold
    gksize : int
        kernel size of gaussian filter
    
    """
    filterd_img = cv2.GaussianBlur(img, (gksize, gksize), 0)
    edges = cv2.Canny(filterd_img, thres1, thres2)
    return edges

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    # read in gray mode
    img = cv2.imread("lenna.png", 0)
    # api of cv2 method
    edges = canny_api(img, 50, 150)
    cv2.imshow("canny by api", edges)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()

    # detail method
    canny = Canny(img.astype(int), 0.6, 3, 50, 130)
    img_edges = canny.detecd_img
    plt.imshow(img_edges, cmap='gray')
    plt.show()
