import cv2
import numpy as np
from imgToGray import img_gray,img_read

def img_binary(imgpath,threshold,debug=0):
    img=img_read(imgpath)
    img_thres=img_gray(img)
    img_thres[img_thres>threshold]=255
    img_thres[img_thres<=threshold]=0
    if debug:
        cv2.imshow('demo1',img_thres)
        cv2.waitKey(0)
    return img_thres

if __name__=="__main__":
    img_binary(imgpath=r"D:\badou_hmwk\158+hesiqing+shanghai\2nd_week\LenaRGB.bmp",threshold=120)
