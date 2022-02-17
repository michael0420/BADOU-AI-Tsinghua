import numpy as np
import cv2

def bilinear(src, size):
    h, w, channel = src.shape
    dest = np.zeros((size[0], size[1], channel), src.dtype)
    scalex = float(size[1]) / w
    scaley = float(size[0]) / h
    for c in range(channel):
        for i in range(size[0]):
            srcy = (i + 0.5) / scaley - 0.5
            y0 = max(0, int(np.floor(srcy)))
            y1 = min(y0+1, h-1)
            for j in range(size[1]):
                srcx = (j + 0.5) / scalex - 0.5
                x0 = max(0, int(np.floor(srcx)))
                x1 = min(x0+1, w-1)
                P1 = (x1 - srcx) * src[y0, x0, c] + (srcx - x0) * src[y0, x1, c]
                P2 = (x1 - srcx) * src[y1, x0, c] + (srcx - x0) * src[y1, x1, c]
                dest[i, j, c] = int((y1 - srcy) * P1 + (srcy - y0) * P2)
    return dest



if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    dest = bilinear(img, (900, 900))
    cv2.imshow("origin", img)
    cv2.imshow("bilinear", dest)
    cv2.waitKey()


