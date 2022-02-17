import numpy as np
import cv2
# import sys
# sys.path.append('/home/uers/PycharmProjects/BADOU-AI-Tsinghua/74-张志嵩-北京')
# from week-1.Img2GrayandBinary import Colourconverter

class ImgRead():
    def __init__(self):
        self.img = cv2.imread('/home/uers/desk_B/八斗/week2/lenna.png')
        self.shape = self.img.shape[:2]
    def img2gray(self):
        h,w = self.shape[:2]
        self.gray = np.zeros((h,w),self.img.dtype)
        for i in range(h):
            for j in range(w):
                self.gray[i][j] = int(self.img[i,j,0]*0.11 + self.img[i,j,1]*0.59 + self.img[i,j,2]*0.3)
        return self.gray

class Histogram_Equalization(ImgRead):
    def __init__(self):
        super(Histogram_Equalization,self).__init__()

    def hist_equalization(self, if_gray=True):
        if if_gray:
            self.img2gray()
            self.hist = cv2.calcHist([self.gray],[0], None, [256],[0,256])
            h, w = self.shape[:2]
            self.gray_histequal = np.zeros((h,w), np.uint8)
            for i in range(h):
                for j in range(w):
                   self.gray_histequal[i][j] = int(self.find_hist(self.gray[i][j])*256/(h*w) - 1)
        else:
            img_split = cv2.split(self.img)
            h, w = self.shape[:2]
            for c in range(3):
                self.hist = cv2.calcHist([img_split[c]],[0], None, [256], [0,256])
                for i in range(h):
                    for j in range(w):
                        img_split[c][i][j] = int(self.find_hist(img_split[c][i][j])*256/(h*w)-1)
            self.color_histequal = cv2.merge((img_split[0],img_split[1],img_split[2]))
    def find_hist(self, k):
        num = 0
        for i in range(k):
            num += self.hist[i]
        return num
    def plot_figure(self):
        import matplotlib.pyplot as plt
        img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(img)
        print("---image lenna----")
        plt.subplot(2,2,2)
        plt.imshow(self.gray,cmap='gray')
        print("---image gray----")
        plt.subplot(2,2,3)
        plt.imshow(self.gray_histequal,cmap='gray')
        print("-----imge_gray_histequal------")
        plt.subplot(2, 2, 4)
        self.color_histequal = cv2.cvtColor(self.color_histequal,cv2.COLOR_BGR2RGB)
        plt.imshow(self.color_histequal)
        print("-----imge_color_histequal------")
        plt.suptitle("Histogram_Equalization")
        plt.show()

if __name__ == "__main__":
    HE = Histogram_Equalization()
    HE.hist_equalization()
    HE.hist_equalization(False)
    HE.plot_figure()