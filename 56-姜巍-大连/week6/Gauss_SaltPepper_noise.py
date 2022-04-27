import numpy as np
import matplotlib.pyplot as plt
import cv2


class NoiseMaker:
    """向图片增加特定噪声"""

    def __init__(self, img):
        """属性初始化"""

        self.img = img.astype(np.float64)
        print(f"图像高(h)：{self.img.shape[0]};\t图像宽(w)：{self.img.shape[1]}。")
        self.noise_mask = np.zeros((1, 1))

    def set_noise_mask(self, size, mode='ga', mean=0.0, sigma=1.0):
        """
        重新设置加噪声数组的尺寸并生成高斯或椒盐噪声
        size: 噪声矩阵的(h, w)
        mode: 传入‘ga’则生成高斯噪声矩阵，传入‘sp’则生成椒盐噪声矩阵
        """

        if mode == 'ga':
            print("您选择添加'高斯噪声'。")
            self.noise_mask = np.random.normal(mean, sigma, size)
        elif mode == 'sp':
            print("您选择添加'椒盐噪声'。")
            self.noise_mask = np.random.randint(-1, 2, size=size)
            self.noise_mask *= 255
            self.noise_mask = np.where(self.noise_mask < 0, -255, 255)
        else:
            print(f"I'm afraid I couldn't recognize {mode}.")

    def get_noised_img(self, mode='rectangle', init_cor=(0, 0)):
        """
        添加噪声
        mode == 'rectangle'，则按照noise_mask的size为img添加一个矩形区域的噪声，此时init_cor参数规定了noise_mask起始顶点的位置；
        mode == 'random'，则将noise_mask变换为img相同尺寸，并用0填充，之后进行打乱array中元素顺序，再为img添加噪声。
        """

        if mode == 'rectangle':
            print("您选择局部矩形区域添加噪声模式。")
            if self.noise_mask.shape[0] + init_cor[0] <= self.img.shape[0] and self.noise_mask.shape[1] + init_cor[1] <= \
                    self.img.shape[1]:
                self.img[init_cor[0]:init_cor[0] + self.noise_mask.shape[0], init_cor[1]:init_cor[1] + self.noise_mask.shape[1]] += self.noise_mask
                self.img = np.where(self.img > 255, 255, np.where(self.img < 0, 0, self.img)).astype(np.uint8)
            else:
                print("Need to reset the 'init_cor'.")
        elif mode == 'random':
            print("您选择全图随机位置添加噪声模式。")
            temp_arr01 = np.zeros(self.img.shape)
            temp_arr01[:self.noise_mask.shape[0], :self.noise_mask.shape[1]] += self.noise_mask
            temp_arr02 = temp_arr01.flatten().T
            np.random.shuffle(temp_arr02)
            temp_arr02 = temp_arr02.reshape(self.img.shape)
            self.img += temp_arr02
            self.img = np.where(self.img > 255, 255, np.where(self.img < 0, 0, self.img)).astype(np.uint8)
        else:
            print(f"I'm afraid I couldn't recognize {mode}.")


image = cv2.imread('lenna.png', 0)  # 读取图像

# 对矩形区域加高斯噪声
noised_img = NoiseMaker(image)
noised_img.set_noise_mask((256, 512), mode='ga', mean=20, sigma=10)
noised_img.get_noised_img(mode='rectangle', init_cor=(255, 0))

# 对随机区域加椒盐噪声
noised_img02 = NoiseMaker(image)
noised_img02.set_noise_mask((128, 128), mode='sp')
noised_img02.get_noised_img(mode='random')

# 输出图像
img02 = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)
cv2.imshow('origin', img2)
cv2.imshow('lenna_GaussianNoise', noised_img.img)
cv2.imshow('lenna_Salt&PepperNoise', noised_img02.img)
cv2.waitKey(0)
