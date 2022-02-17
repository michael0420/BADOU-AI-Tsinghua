import cv2
import numpy as np

def nearest(image_path,u_h,u_w):
    img = cv2.imread(image_path)
    h,w,c = img.shape
    img_n = np.zeros([u_h,u_w,c],np.uint8)
    h_n = u_h/h
    w_n = u_w/w
    for i in range(u_h):
        for j in range(u_w):
            img_n[i,j] = img[int(i/h_n),int(j/w_n)]
    cv2.imshow("image nearest", img_n)
    cv2.waitKey(0)
    print("image nearest", img_n)

if __name__ == '__main__':
    image_path = 'lenna.png'
    u_h = 800
    u_w = 800
    nearest(image_path, u_h, u_w)

