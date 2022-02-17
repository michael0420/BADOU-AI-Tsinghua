import cv2
import numpy as np
import matplotlib.pyplot as plt

img_loc = r"02_math,graph_basic/lenna.png"


class ImageBasic:
    def __init__(self) -> None:
        pass


class ImageProcess(ImageBasic):
    def __init__(self, img_loc):
        super().__init__()
        self.__img = cv2.imread(img_loc)
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

    def NNInterp(self, height_st, width_st):
        img = self.__EmptyImgGen(height=height_st, width=width_st)
        h_zoom_size = height_st / self.__h
        w_zoom_size = width_st / self.__w

        for i in range(height_st):
            for j in range(width_st):
                coord_i = round(i / h_zoom_size)  # reloc coordinates
                coord_j = round(j / w_zoom_size)
                img[i, j] = self.__img[coord_i, coord_j]

        return img


def Main():
    improcess = ImageProcess(img_loc)
    img_raw = improcess.img
    img_enlarge = improcess.NNInterp(800, 800)

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title("origin lenna\n512x512")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_enlarge, cv2.COLOR_BGR2RGB))
    plt.title("enlarge lenna\n800x800")

    plt.suptitle("Nearest Neighbor Interpolation")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    Main()
