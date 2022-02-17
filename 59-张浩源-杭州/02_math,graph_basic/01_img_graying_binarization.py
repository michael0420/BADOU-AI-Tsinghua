import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img_loc = r"02_math,graph_basic/lenna.png"


class ImageBasic:
    def __init__(self) -> None:
        pass


class ImageProcess(ImageBasic):
    def __init__(self, img_loc):
        super().__init__()
        self.__img = cv2.imread(img_loc)  # imread BGR
        self.__h = self.__img.shape[0]
        self.__w = self.__img.shape[1]
        self.__c = self.__img.shape[2]

    @property
    def img(self):
        return self.__img

    def __EmptyImgGen(self, height, width, channel=None, type_=None):
        channel_st = self.__c if not channel else channel
        type_st = self.__img.dtype if not type_ else type_
        img = np.zeros(shape=(height, width, channel_st), dtype=type_st)
        return img

    def GrayingFormular(self):
        img = self.__EmptyImgGen(height=self.__h, width=self.__w, channel=1)
        for i in range(self.__h):
            for j in range(self.__w):
                pixel = self.__img[i, j]
                gray_scale = pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3
                img[i, j] = int(gray_scale)
        return img

    def GrayingModule(self):
        img = rgb2gray(self.__img)
        return img

    def BinariFormular(self):
        img_g = self.GrayingFormular()
        img_b = self.__EmptyImgGen(height=self.__h, width=self.__w, channel=1)
        for i in range(self.__h):
            for j in range(self.__w):
                pixel = img_g[i, j]
                scale = 0 if pixel < 128 else 1
                img_b[i, j] = scale
        return img_b

    def BinariModule(self):
        img_g = self.GrayingModule()
        img_b = np.where(img_g < 0.5, 0, 1)
        return img_b


def Main():
    improcess = ImageProcess(img_loc)
    img_raw = improcess.img
    img_gray_0 = improcess.GrayingFormular()
    img_gray_1 = improcess.GrayingModule()
    img_bin_0 = improcess.BinariFormular()
    img_bin_1 = improcess.BinariModule()

    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title("origin lenna")

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(img_gray_0, cmap="gray")
    plt.title("graying lenna")

    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(img_bin_0, cmap="gray")
    plt.title("binarization lenna")

    plt.suptitle("Image Processing Basics")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    Main()
