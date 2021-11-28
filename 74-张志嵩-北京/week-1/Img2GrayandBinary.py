import numpy as np
import cv2

class Colourconverter():
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
    def img2binary(self):
        h, w = self.shape[:2]
        self.img2gray()
        self.binary = np.zeros((h,w),self.img.dtype)
        for i in range(h):
            for j in range(w):
                if self.gray[i,j] > 128:
                    self.binary[i,j] = 255
                else:
                    self.binary[i,j] = 0
        return self.binary

    def plot_figure(self):
        import matplotlib.pyplot as plt
        img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        plt.subplot(131)
        plt.imshow(img)
        print("---image lenna----")
        plt.subplot(1,3,2)
        plt.imshow(self.gray,cmap='gray')
        print("---image gray----")
        plt.subplot(1,3,3)
        plt.imshow(self.binary,cmap='gray')
        print("-----imge_binary------")
        plt.show()



if __name__ == "__main__":
    print('I am happy!')
    #img = cv2.imread('/home/uers/desk_B/八斗/week2/lenna.png')
    Clter = Colourconverter()
    gray = Clter.img2gray()
    binary = Clter.img2binary()
    Clter.plot_figure()
    # cv2.imshow('figure',gray)
    # cv2.imshow('figure2', binary)
    # cv2.waitKey(0)