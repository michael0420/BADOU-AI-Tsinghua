import numpy as np
import cv2
from Img2GrayandBinary import Colourconverter

class Nearest(Colourconverter):
    def __init__(self, h,w,c):
        super(Nearest, self).__init__()
        self.h = h
        self.w = w
        self.zoom_n = np.zeros((h,w,c), np.uint8)

    def nearest(self):
        h, w= self.shape[:2]
        delth = self.h/h
        deltw = self.w/w
        for i in range(self.h):
            for j in range(self.w):
                self.zoom_n[i,j] = self.img[round(i/delth),round(j/deltw)]
        return self.zoom_n
    def plot_figure(self):
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        print("---image lenna----")
        plt.subplot(1,2,2)
        img_n = cv2.cvtColor(self.zoom_n, cv2.COLOR_BGR2RGB)
        plt.imshow(img_n)
        print("---zoom_n lenna----")
        plt.show()
    def cv_figure(self):
        cv2.imshow('oral', self.img)
        print("---image lenna----")
        cv2.imshow('zoom_n',self.zoom_n)
        print("---zoom_n lenna----")
        cv2.waitKey(0)

if __name__ == "__main__":
    Nst = Nearest(800,800,3)
    Nst.nearest()
    Nst.plot_figure()
    #Nst.cv_figure()


