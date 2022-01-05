import cv2
import numpy as np


def gauss_noise(img,mu=0,sigma=0.05):
    gs_mat=np.random.normal(mu,sigma,img.shape)
    img_f=np.float64(img/255)
    # if len(img.shape)==3:
    #     gs_noiseimg=img_f+np.dstack((gs_mat,gs_mat,gs_mat))
    # else:
    gs_noiseimg=img_f+gs_mat
    output=np.uint8(gs_noiseimg*255)
    if True:
        combine=np.hstack((img,output))
        cv2.imshow('gaussnoise',combine)
        cv2.waitKey(0)
    return output

def salt_noise(img,ratio=0.05):
    salt_img=img.copy()
    #纯黑或白噪点
    # salt_mat=np.random.choice([0,1],size=img.shape,p=(1-ratio,ratio))
    # # salt_img[salt_mat==1]=np.random.choice([0,255])
    # salt_img[salt_mat[:,:,0]==1,:]=np.random.choice([0,255])
    
    #黑白都有
    salt_mat=np.random.choice([0,1,2],size=img.shape,p=(1-ratio,ratio/2,ratio/2))
    #此种方式三通道带彩色效果
    salt_img[salt_mat==1]=np.random.choice([0,255])
    salt_img[salt_mat==2]=np.random.choice([0,255])
    #纯黑白噪点
    # salt_img[salt_mat[:,:,0]==1,:]=255
    # salt_img[salt_mat[:,:,0]==2,:]=0
    # np.random.choice([0,255])
    if True:
        combine=np.hstack((img,salt_img))
        cv2.imshow('salt',combine)
        cv2.waitKey(0)
    return salt_img

if __name__=="__main__":
    img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\6th_week\LenaRGB.bmp"
    img=cv2.imread(img_path,-1)
    # gauss_noise(img)
    salt_noise(img)