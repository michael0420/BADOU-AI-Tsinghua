import cv2
import numpy
import random

class Img_relative():
    def __init__(self, a=None):
        self.img = cv2.imread('/home/uers/desk_B/八斗/week2/lenna.png')
        self.shape = self.img.shape[:2]

class gauss_noise(Img_relative):
    def __init__(self, mean=30, sigma=60, percentage=0.8,b=None):
        super(gauss_noise,self).__init__(a=b)
        self.mean = mean
        self.sigma = sigma
        self.percentage = percentage

    def addNoise(self):
        NoiseImg = self.img.copy()
        h, w = self.shape
        num = int(self.percentage*h*w)
        for i in range(num):
            randX = random.randint(0,h-1)
            randY = random.randint(0,w-1)
            for j in range(3):
                NoiseImg[randX,randY,j] = self.img[randX,randY,j] + random.gauss(self.mean,self.sigma)
                if NoiseImg[randX,randY,j] <0:
                    NoiseImg[randX,randY,j] = 0
                elif NoiseImg[randX,randY,j] > 255:
                    NoiseImg[randX,randY,j] = 255
        return NoiseImg
    def plot_img(self, NoiseImg):
        img2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("orl", self.img)
        noise = cv2.cvtColor(NoiseImg, cv2.COLOR_BGR2GRAY)
        cv2.imshow("GaussianNoise", NoiseImg)
        cv2.waitKey(0)

if __name__ == "__main__":
    GN = gauss_noise(30,60,0.8)
    NoiseImg = GN.addNoise()
    GN.plot_img(NoiseImg)