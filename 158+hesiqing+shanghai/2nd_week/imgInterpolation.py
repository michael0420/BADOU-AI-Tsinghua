import cv2
import numpy as np


def bilinear(dst_x,dst_y,src_shape,dst_shape,src_img):
    src_w,src_h=src_shape[0:2]
    dst_w,dst_h=dst_shape[0:2]
    src_x=(dst_x+0.5)*(src_w/dst_w)-0.5
    src_y=(dst_y+0.5)*(src_h/dst_h)-0.5
    src_xi=int(src_x)
    src_yi=int(src_y)
    x01=src_x-src_xi
    x02=src_xi-src_x+1
    y01=src_y-src_yi
    y02=src_yi+1-src_y
    if src_xi+1==src_w or src_yi+1==src_h:
        pix_dst=src_img[src_xi,src_yi]
    else:
        pix_dst=x02*y02*src_img[src_xi,src_yi]+x01*y02*src_img[src_xi+1,src_yi]+x02*y01*src_img[src_xi,src_yi+1]+x01*y01*src_img[src_xi+1,src_yi+1]
    return pix_dst

def nearest(dst_x,dst_y,src_shape,dst_shape,src_img):
    src_w,src_h=src_shape[0:2]
    dst_w,dst_h=dst_shape[0:2]
    src_x=(dst_x+0.5)*(src_w/dst_w)-0.5
    src_y=(dst_y+0.5)*(src_h/dst_h)-0.5
    src_xi=int(src_x)
    src_yi=int(src_y)
    pix_dst=src_img[src_xi,src_yi]
    return pix_dst

#cv2.remap
def img_interpolation(img_path,dst_shape,insert_type="bilinear",debug=1):
    img=cv2.imread(img_path,-1)
    dst_img=np.zeros(dst_shape,dtype=np.uint8)
    if insert_type=="bilinear":
        for channel in range(img.shape[2]):
            for x in range(dst_img.shape[0]):
                for y in range(dst_img.shape[1]):
                    dst_img[x,y,channel]=bilinear(x,y,src_shape=img.shape,dst_shape=dst_shape,src_img=img[:,:,channel])
    elif insert_type=="nearest":
        for channel in range(img.shape[2]):
            for x in range(dst_img.shape[0]):
                for y in range(dst_img.shape[1]):
                    dst_img[x,y,channel]=nearest(x,y,src_shape=img.shape,dst_shape=dst_shape,src_img=img[:,:,channel])
    else:
        raise Exception('unexpected insert type')
    if debug==1:
        cv2.imshow('demo2',dst_img)
        cv2.waitKey(0)
    pass

if __name__=="__main__":
    img_interpolation(img_path=r"D:\badou_hmwk\158+hesiqing+shanghai\2nd_week\LenaRGB.bmp",
    dst_shape=(600,700,3),insert_type="bilinear",debug=1)