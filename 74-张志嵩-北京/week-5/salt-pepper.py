import cv2
import numpy
import random
from gauss_noise import Img_relative, gauss_noise

class saltPepper(gauss_noise):
    def __init__(self, percentage=0.2):
        super(saltPepper,self).__init__(percentage=percentage)

    def addSalt(self):
        h,w = self.shape
        Salt = self.img.copy()
        num = int(self.percentage*h*w)
        for i in range(num):
            randX = random.randint(0, h-1)
            randY = random.randint(0, w-1)
            if random.random() < 0.5:
                Salt[randX, randY, :] = 0
            else:
                Salt[randX, randY, :] = 255
        return Salt

    def plot_img(self, Salt, NoiseImg=None):
        cv2.imshow("orl", self.img)
        if NoiseImg.any():
            cv2.imshow("NoiseImg", NoiseImg)
        cv2.imshow("SaltImg", Salt)
        cv2.waitKey(0)


if __name__ == "__main__":
    sP = saltPepper(0.2)
    Noise = sP.addNoise()
    Salt = sP.addSalt()
    sP.plot_img(Salt, Noise)
