import cv2
import numpy as np
from tqdm import tqdm
# from scipy.signal import convolve

img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\4th_week\LenaRGB.bmp"
grey_img=cv2.imread(img_path,0)

def canny(img):
    pass

def conv(img,kernal,mode,stride=1):
    assert mode in ['valid','same','full']
    h,w=img.shape[0:2]
    h_k,w_k=kernal.shape[0:2]
    kernal_rot=kernal[::-1,::-1]
    if mode=="same":
        h_pad=h_k//2
        w_pad=w_k//2
        h_new,w_new=h+2*h_pad,w+2*w_pad
        pad_img=np.zeros((h_new,w_new))
        pad_img[h_pad:h_pad+h,w_pad:w+w_pad]=img
        tg_h,tg_w=(h+2*h_pad-h_k)//stride+1,(w+2*w_pad-w_k)//stride+1
        result_img=np.zeros((tg_h,tg_w),np.uint8)
        for i in range(0,h_new-h_k,stride):
            for j in range(0,w_new-w_k,stride):
                con_part=pad_img[i:i+h_k,j:j+w_k]
                sum_ij=np.sum(con_part*kernal_rot)
                result_img[i,j]=min(max(sum_ij,0),255)
        return  result_img
def sobel(img,debug=1):
    kernal_x=np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
    kernal_y=np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]])
    convolve_img=conv(img,kernal_x,mode='same')
    cv_covimg=cv2.filter2D(img,-1,kernal_x[::-1,::-1])
    sobel_cvx=cv2.convertScaleAbs(cv2.Sobel(img,cv2.CV_16S,1,0))
    if debug:
        combine=np.hstack((cv_covimg,convolve_img))
        cv2.imshow('sobel',combine)
        cv2.waitKey(0)
    return convolve_img

def prewitt():
    #都合在edge_pre中
    pass

def edge_pre(img,edge_kind,debug=0):
    assert edge_kind in ['soble_x','soble_y','prewitt_x','prewitt_y']
    kernel_dict={'soble_x':np.array([[-1,0,1],
                                     [-2,0,2],
                                     [-1,0,1]]),
                 'soble_y':np.array([[-1,-2,-1],
                                     [0,0,0],
                                     [1,2,1]]),
                 'prewitt_x':np.array([[-1,0,1],
                                       [-1,0,1],
                                       [-1,0,1]]),
                 'prewitt_y':np.array([[1,1,1],
                                       [0,0,0],
                                       [-1,-1,-1]])}
    img_blur=cv2.GaussianBlur(img,(3,3),0)
    convolve_img=conv(img_blur,kernal=kernel_dict[edge_kind],mode='same')
    if debug:
        # combine=np.hstack((cv_covimg,convolve_img))
        cv2.imshow('sobel',convolve_img)
        cv2.waitKey(0)
    return convolve_img

def nms(img_grey):
    """non maximum suppression
    """
    grad_x=edge_pre(img_grey,edge_kind='prewitt_x')
    grad_y=edge_pre(img_grey,edge_kind='prewitt_y')
    grad=np.sqrt(grad_x**2+grad_y**2)
    grad_x[grad_x==0]==1e-7
    tan_mat=grad_y/grad_x
    h_ta,w_ta=tan_mat.shape[0:2]
    nms_result=np.zeros(img_grey.shape)
    for i in range(1,h_ta-1):
        for j in range(1,w_ta-1):
            flag=True
            neib_mat=tan_mat[i-1:i+2,j-1:j+2]
            if tan_mat[i,j]<-1:
                thres1=(1+1/tan_mat[i,j])*neib_mat[0,1]-1/tan_mat[i,j]*neib_mat[0,0]
                thres2=(1+1/tan_mat[i,j])*neib_mat[2,1]-1/tan_mat[i,j]*neib_mat[2,2]
            elif tan_mat[i,j]>1:
                thres1=(1-1/tan_mat[i,j])*neib_mat[0,1]+1/tan_mat[i,j]*neib_mat[0,2]
                thres2=(1-1/tan_mat[i,j])*neib_mat[2,1]+1/tan_mat[i,j]*neib_mat[2,0]
            elif tan_mat[i,j]>=0:
                thres1=(1-tan_mat[i,j])*neib_mat[1,2]+tan_mat[i,j]*neib_mat[0,2]
                thres2=(1-tan_mat[i,j])*neib_mat[1,0]+tan_mat[i,j]*neib_mat[2,0]
            else:
                thres1=(1+tan_mat[i,j])*neib_mat[1,2]-tan_mat[i,j]*neib_mat[2,2]
                thres2=(1+tan_mat[i,j])*neib_mat[1,0]-tan_mat[i,j]*neib_mat[0,0]
            if not grad[i,j]>max(thres1,thres2):
                flag=False
            if flag:
                nms_result[i,j]=grad[i,j]
    return nms_result
if __name__=="__main__":
    # img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\4th_week\LenaRGB.bmp"
    # grey_img=cv2.imread(img_path,0)
    sobel(img=grey_img)