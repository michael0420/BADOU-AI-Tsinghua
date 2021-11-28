import cv2
import numpy as np

def linear(image_path,h_n,w_n):
    img = cv2.imread(image_path)
    h,w,c = img.shape
    img_lin = np.zeros((h_n, w_n, c), dtype=np.uint8)
    h_r = h/h_n
    w_r = w/w_n
    for i in range(c):
        for y_n in range(h_n):
            for x_n in range(w_n):
                x = (x_n + 0.5) * w_r - 0.5
                y = (y_n + 0.5) * h_r - 0.5

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, w - 1)

                y0 = int(np.floor(y))
                y1 = min(y0 + 1, h - 1)

                temp0 = (x1 - x) * img[y0, x0, i] + (x - x0) * img[y0, x1, i]
                temp1 = (x1 - x) * img[y1, x0, i] + (x - x0) * img[y1, x1, i]
                img_lin[y_n, x_n, i] = (y1 - y) * temp0 + (y - y0) * temp1
    cv2.imshow("image linear", img_lin)
    cv2.waitKey(0)
    print("image linear", img_lin)

if __name__ == '__main__':
    image_path = 'lenna.png'
    h_n = 800
    w_n = 800
    linear(image_path, h_n, w_n)