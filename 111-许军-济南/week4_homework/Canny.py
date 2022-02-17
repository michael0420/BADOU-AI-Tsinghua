# -- coding:utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt
class Canny:
    def __init__(self,gray_img):
        self.img = gray_img
        self.k_size = 3
        self.sigma = 0.5
        self.pad = self.k_size // 2
        self.h,self.w = self.img.shape
        self.gaus_img = self._gaussfilter()
        self.sobelx = cv2.Sobel(self.gaus_img,cv2.CV_64F,1,0,ksize=3)
        self.sobely = cv2.Sobel(self.gaus_img,cv2.CV_64F,0,1,ksize=3)
        self.sobel_img  = cv2.Sobel(self.gaus_img,cv2.CV_64F,1,1,ksize=3)
        self.sobelx[self.sobelx == 0] = 0.0000001
        self.sobely[self.sobely == 0] = 0.0000001
        self.angle = self.sobely / self.sobelx
        self.nms_image = self._nms()
        self.dual_image = self._dualthreshold()

    def _gaussfilter(self):
        tar_img = np.zeros((self.h+self.pad*2,self.w+self.pad*2),np.uint8)
        tar_img[self.pad:self.h+self.pad,self.pad:self.w+self.pad] = self.img.copy().astype(np.uint8)
        kernel = np.zeros((self.k_size,self.k_size),np.float64)
        for x in range(-self.pad,-self.pad+self.k_size):
            for y in range(-self.pad,-self.pad+self.k_size):
                kernel[x+self.pad][y+self.pad] = np.exp(-(x**2+y**2)/2*(self.sigma**2))
        kernel = kernel/(np.pi*2*self.sigma**2)
        kernel /= kernel.sum()
        tmp = tar_img.copy()
        for y in range(self.h):
            for x in range(self.w):
                tar_img[y+self.pad,x+self.pad] = np.sum(kernel*tmp[y:y+self.k_size,x:x+self.k_size])
        tar_img = np.clip(tar_img,0,255)
        tar_img = tar_img[self.pad:self.pad+self.h,self.pad:self.pad+self.w].astype(np.uint8)
        return tar_img
    # 非极大值抑制
    def _nms(self):
        nms_img = np.zeros(self.img.shape)
        for i in range(self.pad,self.h-self.pad):
            for j in range(self.pad,self.w-self.pad):
                flag = True
                tmp = self.sobel_img[i-self.pad:i+self.pad+1,j-self.pad:j+self.pad+1]
                if self.angle[i,j]>=1:
                    num1 = tmp[0,1] + (tmp[0,2] - tmp[0,1])/self.angle[i,j]
                    num2 = tmp[2,1] + (tmp[2,0] - tmp[2,1])/self.angle[i,j]
                    if not (self.sobel_img[i,j] > num2 and self.sobel_img[i,j] > num1):
                        flag = False
                elif self.angle[i, j] > 0 :
                    num_1=(tmp[0, 2] - tmp[1, 2]) * self.sobel_img[i, j] + tmp[1, 2]
                    num_2=(tmp[2, 0] - tmp[1, 0]) * self.angle[i, j] + tmp[1, 0]
                    if not (self.sobel_img[i, j] > num_1 and self.sobel_img[i, j] > num_2) :
                        flag=False
                elif self.angle[i, j] < 0 :
                    num_1=(tmp[1, 0] - tmp[0, 0]) * self.angle[i, j] + tmp[1, 0]
                    num_2=(tmp[1, 2] - tmp[2, 2]) * self.angle[i, j] + tmp[1, 2]
                    if not (self.sobel_img[i, j] > num_1 and self.sobel_img[i, j] > num_2) :
                        flag=False
                else:
                    num_1=(tmp[0, 1] - tmp[0, 0]) / self.angle[i, j] + tmp[0, 1]
                    num_2=(tmp[2, 1] - tmp[2, 2]) / self.angle[i, j] + tmp[2, 1]
                    if not (self.angle[i, j] > num_1 and self.angle[i, j] > num_2) :
                        flag=False
                if flag:
                    nms_img[i,j] = self.sobel_img[i,j]
        return nms_img
    # 双阈值
    def _dualthreshold(self):
        lower_boundary=self.sobel_img.mean() * 0.5
        high_boundary=lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
        zhan=[]
        for i in range(1, self.nms_image.shape[0] - 1) :  # 外圈不考虑了
            for j in range(1, self.nms_image.shape[1] - 1) :
                if self.nms_image[i, j] >= high_boundary :  # 取，一定是边的点
                    self.nms_image[i, j]=255
                    zhan.append([i, j])
                elif self.nms_image[i, j] <= lower_boundary :  # 舍
                    self.nms_image[i, j]=0

        while not len(zhan) == 0 :
            temp_1, temp_2=zhan.pop()  # 出栈
            a=self.nms_image[temp_1 - 1 :temp_1 + 2, temp_2 - 1 :temp_2 + 2]
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary) :
                self.nms_image[temp_1 - 1, temp_2 - 1]=255  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary) :
                self.nms_image[temp_1 - 1, temp_2]=255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary) :
                self.nms_image[temp_1 - 1, temp_2 + 1]=255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary) :
                self.nms_image[temp_1, temp_2 - 1]=255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary) :
                self.nms_image[temp_1, temp_2 + 1]=255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary) :
                self.nms_image[temp_1 + 1, temp_2 - 1]=255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary) :
                self.nms_image[temp_1 + 1, temp_2]=255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary) :
                self.nms_image[temp_1 + 1, temp_2 + 1]=255
                zhan.append([temp_1 + 1, temp_2 + 1])

        for i in range(self.nms_image.shape[0]) :
            for j in range(self.nms_image.shape[1]) :
                if self.nms_image[i, j] != 0 and self.nms_image[i, j] != 255 :
                    self.nms_image[i, j]=0
        return self.nms_image

if __name__ == '__main__':
    img = cv2.imread("img/lenna.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    p = Canny(img)
    out = p._gaussfilter()
    plt.figure()
    plt.subplot(241)
    plt.title("原图")
    plt.imshow(img,cmap="gray")
    plt.subplot(242)
    plt.title("高斯滤波")
    plt.imshow(p.gaus_img,cmap="gray")
    plt.subplot(243)
    plt.title("x边缘")
    plt.imshow(p.sobelx,cmap="gray")
    plt.subplot(244)
    plt.title("y边缘")
    plt.imshow(p.sobely,cmap="gray")
    plt.subplot(245)
    plt.title("xy边缘")
    plt.imshow(p.sobel_img,cmap="gray")
    plt.subplot(246)
    plt.title("极大抑制图")
    plt.imshow(p.nms_image,cmap="gray")
    plt.subplot(247)
    plt.imshow(p.dual_image)
    plt.show()



