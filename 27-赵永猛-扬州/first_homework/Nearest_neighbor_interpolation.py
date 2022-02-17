import numpy as np
import cv2


def nearest_nei_inter(image, height, width):
    src_height, src_width, channels = image.shape
    empty_img = np.zeros((height, width, channels), np.uint8)
    for k in range(channels):
        for i in range(height):
            for j in range(width):
                empty_img[i, j, k] = image[int(i*src_height/height), int(j*src_width/width), k]
    return empty_img


if __name__ == '__main__':
    img = cv2.imread("11-1.png")
    out = nearest_nei_inter(img, 500, 500)
    cv2.imshow("image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
