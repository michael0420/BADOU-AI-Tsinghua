import cv2
import numpy as np

def trans(img,p1,p2,debug=True):
    h,w=img.shape[0:2]
    M = cv2.getPerspectiveTransform(p1,p2)
    tg_res=cv2.warpPerspective(img,M,(w,h))
    if debug:
        combine=np.hstack((img,tg_res))
        cv2.imshow('perspective',combine)
        cv2.waitKey(0)
    return tg_res

if __name__=="__main__":
    img_path=r'D:\badou_hmwk\158+hesiqing+shanghai\4th_week\LenaRGB.bmp'
    img=cv2.imread(img_path,-1)
    h,w=img.shape[0:2]
    p1=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    p2=np.float32([[int(w/4),int(h/4)],[int((w-1)*3/4),0],[w-1,h-1],[0,h-1]])
    trans_img=trans(img,p1=p1,
                    p2=p2)
    trans_back=trans(trans_img,p2,p1)
    print(np.all(trans_back==img))