import numpy as np
import cv2
from Nearest import Nearest

class Bilinear(Nearest):
    def __init__(self,h,w,c):
        super(Bilinear,self).__init__(h,w,c)
        self.h = h
        self.w = w
        self.zoom_b = np.zeros((h, w, c), np.uint8)

    def bilinear(self):
        h, w = self.shape[:2]
        delth = self.h / h
        deltw = self.w / w
        for i in range(self.h):
            for j in range(self.w):
                m = int((i+0.5)/delth-0.5)
                n = int((j+0.5)/deltw-0.5)
                u = i/delth - m
                v = j/deltw - n
                self.zoom_b[i,j] = (1-u)*(1-v)*self.img[m,n] + (1-u)*v*self.img[m,min(n+1, w-1)] + u*(1-v)*self.img[min(m+1,h-1),n] + u*v*self.img[min(m+1,h-1),min(n+1,w-1)]
        return self.zoom_b

    def plot_figure(self):
        import matplotlib.pyplot as plt
        self.nearest()
        plt.subplot(1, 3, 1)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        print("---image lenna----")
        plt.subplot(1, 3, 2)
        img_n = cv2.cvtColor(self.zoom_n, cv2.COLOR_BGR2RGB)
        plt.imshow(img_n)
        print("---zoom_n lenna----")
        plt.subplot(1, 3, 3)
        img_b = cv2.cvtColor(self.zoom_b, cv2.COLOR_BGR2RGB)
        plt.imshow(img_b)
        print("---zoom_b lenna----")
        plt.show()
    def cv_figure(self):
        self.nearest()
        cv2.imshow('oral', self.img)
        print("---image lenna----")
        cv2.imshow('zoom_n',self.zoom_n)
        print("---zoom_n lenna----")
        cv2.imshow('zoom_b', self.zoom_b)
        print("---zoom_b lenna----")
        cv2.waitKey(0)

if __name__ == "__main__":
    Bln = Bilinear(800,800,3)
    Bln.bilinear()
    #Bln.plot_figure()
    Bln.cv_figure()