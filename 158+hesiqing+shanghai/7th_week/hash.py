import cv2
import numpy as np

def mean_hash(img,hash_shape=(8,8)):
    h_hs,w_hs=hash_shape[0:2]
    img_reshape=cv2.resize(img,hash_shape)
    gray_img=cv2.cvtColor(img_reshape,cv2.COLOR_RGB2GRAY)
    mn_pix=np.mean(gray_img)
    hs_result=[]
    for i in range(h_hs):
        for j in range(w_hs):
            if gray_img[i][j]>mn_pix:
                hs_result.append(1)
            else:
                hs_result.append(0)
    return hs_result

if __name__=="__main__":
    img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\7th_week\LenaRGB.bmp"
    img=cv2.imread(img_path,-1)
    hs_result=mean_hash(img)
