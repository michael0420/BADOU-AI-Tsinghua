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

    def __Centralization(self, coord_dst, zoom_size):
        coord_src = (coord_dst + 0.5) / zoom_size - 0.5
        return coord_src

    def __NNCoordDetect(self, x, y):
        x_0 = int(np.floor(x))
        x_1 = min(x_0 + 1, self.__w - 1)
        y_0 = int(np.floor(y))
        y_1 = min(y_0 + 1, self.__h - 1)

        return x_0, x_1, y_0, y_1

    def __LinearInterp(self, x, x_0, f_x_0, x_1, f_x_1):
        f_y = (x_1 - x) * f_x_0 + (x - x_0) * f_x_1
        return f_y

    def BilinearInterp(self, height_st, width_st):
        img = self.__EmptyImgGen(height=height_st, width=width_st)
        h_zoom_size = height_st / self.__h
        w_zoom_size = width_st / self.__w

        for c in range(img.shape[2]):
            for i in range(height_st):
                for j in range(width_st):

                    # Set the src coordinates to geometric center
                    x = self.__Centralization(i, h_zoom_size)
                    y = self.__Centralization(j, w_zoom_size)

                    # Locate the nearest four points' coordinates
                    x_0, x_1, y_0, y_1 = self.__NNCoordDetect(x, y)

                    # Three times linearInterpolation get the specify value of the channel
                    r_0 = self.__LinearInterp(
                        x, x_0, self.__img[x_0, y_0, c], x_1, self.__img[x_1, y_0, c]
                    )
                    r_1 = self.__LinearInterp(
                        x, x_0, self.__img[x_0, y_1, c], x_1, self.__img[x_1, y_1, c]
                    )
                    value = self.__LinearInterp(y, y_0, r_0, y_1, r_1)

                    # Assignment
                    img[i, j, c] = round(value)

        return img


def Main():
    improcess = ImageProcess(img_loc)
    img_raw = improcess.img
    img_enlarge = improcess.BilinearInterp(800, 800)

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title("origin lenna\n512x512")

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_enlarge, cv2.COLOR_BGR2RGB))
    plt.title("enlarge lenna\n800x800")

    plt.suptitle("Bilinear Interpolation")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    Main()
