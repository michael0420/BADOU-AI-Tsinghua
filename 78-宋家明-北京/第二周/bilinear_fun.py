import cv2
import numpy as np
import argparse

def bilinear_fun(img, dst_hw):
    dst_h, dst_w = dst_hw
    src_h, src_w, _ = img.shape
    
    dst_img = np.zeros((dst_h,dst_w,3),np.uint8)+255
    size_thh, size_thw = src_h/dst_h, src_w/dst_w
    for c in range(3):
        for ix in range(dst_w):
            for jy in range(dst_h):
                src_x = (ix+0.5)*size_thw-0.5
                src_y = (jy+0.5)*size_thh-0.5
                if src_x+1>src_w:
                    src_x = src_w-1
                if src_y+1>src_h:
                    src_y = src_h-1
                x_re = src_x%1
                y_re = src_y%1
                if x_re==0 and y_re==0:
                    dst_img[jy,ix,c] = img[int(src_y),int(src_x),c]
                elif x_re==0:      
                    dst_img[jy,ix,c] = (img[int(np.ceil(src_y)),int(src_x),c] - img[int(np.floor(src_y)),int(src_x),c])*y_re + img[int(np.floor(src_y)),int(src_x),c]
                elif y_re==0:
                    dst_img[jy,ix,c] = (img[int(src_y),int(np.ceil(src_x)),c] - img[int(src_y),int(np.floor(src_x)),c])*x_re + img[int(src_y),int(np.floor(src_x)),c]
                    
                else :
                    pass   
                    up = (img[int(np.floor(src_y)),int(np.ceil(src_x)),c] - img[int(np.floor(src_y)),int(np.floor(src_x)),c])*x_re + img[int(np.floor(src_y)),int(np.floor(src_x)),c]
                    down = (img[int(np.ceil(src_y)),int(np.ceil(src_x)),c] - img[int(np.ceil(src_y)),int(np.floor(src_x)),c])*x_re + img[int(np.ceil(src_y)),int(np.floor(src_x)),c]
                    dst_img[jy,ix,c] = (down - up)*y_re + down
    dst_img = dst_img.astype(np.uint8)
    
    return dst_img







if __name__=='__main__':
    parser = argparse.ArgumentParser(description='bilinear fun')
    parser.add_argument('--dst_w', type=int, default=800, help='dst_w')
    parser.add_argument('--dst_h', type=int, default=800, help='dst_h')
    parser.add_argument('--img', type=str, default='lenna.png', help='img')
    opt = parser.parse_args()

    img = cv2.imread(opt.img)
    dst_h, dst_w = opt.dst_h, opt.dst_w
    bili_img = bilinear_fun(img, (dst_h, dst_w))

#    cv2.imshow('biliimg',bili_img)
    cv2.imwrite('biliimg.jpg',bili_img)
    cv2.imwrite('biliimg0.jpg',bili_img[:,:,0])
    cv2.imwrite('biliimg1.jpg',bili_img[:,:,1])
    cv2.imwrite('biliimg2.jpg',bili_img[:,:,2])
#    cv2.waitKey(0)
