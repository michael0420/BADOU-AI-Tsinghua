import cv2
import numpy as np

def img_read(img_path):
    img=cv2.imread(img_path,-1)
    return img

def equal_hist(img):
    img_flatten=img.flatten()
    h,w=img.shape[0:2]
    new_img=np.zeros(img.shape,np.uint8)
    vals,counts=np.unique(img_flatten,return_counts=True)
    ac_count=0
    for val,count in zip(vals,counts):
        ac_count+=count/(h*w)
        if ac_count*256-int(ac_count*256)>0.5:
            new_pixel=int(ac_count*256)
        else:
            new_pixel=max(int(ac_count*256)-1,0)
        new_img[img==val]=new_pixel
    return new_img   

def equal_hist_rgb(img_path,debug=1):
    img=img_read(img_path)
    new_img=np.zeros(img.shape,np.uint8)
    new_img[:,:,0]=equal_hist(img[:,:,0])
    new_img[:,:,1]=equal_hist(img[:,:,1])
    new_img[:,:,2]=equal_hist(img[:,:,2])
    b=cv2.equalizeHist(img[:,:,0])
    g=cv2.equalizeHist(img[:,:,1])
    r=cv2.equalizeHist(img[:,:,2])
    api_res=cv2.merge((b,g,r))
    if debug:
        combine=np.hstack((api_res,new_img))
        cv2.imshow("demo",combine)
        cv2.waitKey(0)
if __name__=="__main__":
    pass
    # img=img_read(r"D:\badou_hmwk\158+hesiqing+shanghai\3rd_week\LenaRGB.bmp")
    # aa=equal_hist(img[:,:,0])
    equal_hist_rgb(img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\3rd_week\LenaRGB.bmp",debug=1)