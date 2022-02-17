import numpy as np
import cv2


def img_read(img_path):
    img=cv2.imread(img_path,-1)
    return img
def img_gray(img,debug=0):
    B,G,R=img[:,:,0],img[:,:,1],img[:,:,2]
    # cv2.split(img)
    img_grey=(0.3*R+0.59*G+0.11*B).astype(np.uint8)
    if debug:
        cv2.imshow("demo",img_grey)
        cv2.waitKey(0)
    return img_grey

if __name__=="__main__":
    img=img_read(img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\2nd_week\\LenaRGB.bmp")
    img_gray(img)

