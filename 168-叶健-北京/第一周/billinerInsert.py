import numpy as np
import cv2

"""
双线性插值
"""
def bilinearinsert(imag,outdim):
    src_h, src_w, channel = imag.shape
    print("源图片高:%d,宽:%d"%(src_h, src_w))
    dst_h, dst_w = outdim[1], outdim[0]
    print("目标图片高:%d,宽:%d"%(dst_h, dst_w))
    # 求缩放比例
    scale_h = dst_h/src_h
    scale_w = dst_w/src_w
    print("高度缩放比例是:%f,宽度缩放比例是:%f;"%(scale_h,scale_w))
    if (scale_h==1.0 and scale_w==1.0):
        return imag
    dtype = imag.dtype
    imag_new=np.zeros([dst_h,dst_w,channel],dtype)
    for i in range(channel):
        for h in range(dst_h):
            for w in range(dst_w):
                src_ww=(w+0.5)/scale_w-0.5
                src_hh=(h+0.5)/scale_h-0.5
                src_w1=int(src_ww)
                src_h1=int(src_hh)
                src_w2=min(src_w-1,src_w1+1)
                src_h2=min(src_h-1,src_h1+1)
                tmp_l=(src_w2-src_ww)*imag[src_w1,src_h1,i]+(src_ww-src_w1)*imag[src_w2,src_h1,i]
                tmp_r=(src_w2-src_ww)*imag[src_w1,src_h2,i]+(src_ww-src_w1)*imag[src_w2,src_h2,i]
                imag_new[w,h,i]=tmp_l*(src_h2-src_hh)+tmp_r*(src_hh-src_h1)
    return imag_new




if __name__ == '__main__':
    imag=cv2.imread("lenna.png")
    imag_new=bilinearinsert(imag,(700,700))
    cv2.imshow("src_picture",imag)
    cv2.imshow("dst_picture",imag_new)
    cv2.waitKeyEx()