import cv2
import numpy as np

def extend_image_nearest_insert(img,dst_h,dst_w):
    src_h,src_w,channels=img.shape
    new_image=np.zeros((dst_h,dst_w,channels),np.uint8)
    h_scale=dst_h/src_h
    w_scale=dst_w/src_w
    for i in range(dst_h):
        for j in range(dst_w):
            x=int(i/h_scale)
            y=int(j/w_scale)
            new_image[i,j]=img[x,y]

    return new_image
if __name__ == '__main__':
    img=cv2.imread('lenna.png')
    dst_h, dst_w = 800, 800
    new_image=extend_image_nearest_insert(img, 800, 800)
    cv2.imshow("image",img)
    cv2.imshow("new_image",new_image)
    cv2.waitKey(0)