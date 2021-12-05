import numpy as np
import cv2

def bilinear_interpolation(img, out_dim):
    sou_h, sou_w, channels = img.shape
    dis_h , dis_w = out_dim[1], out_dim[0]
    print('sou_h, sou_w = ', sou_h, sou_w)
    print('dis_h, dis_w = ', dis_h, dis_w)
    if sou_h == dis_h and sou_w == dis_w:
        return img.copy()
    dis_img = np.zeros((dis_h, dis_w , 3), np.uint8)
    scale_x, scale_y = float(sou_w)/dis_w, float(sou_h)/dis_h
    for i in range(3):
        for dis_y in range(dis_h):
            for dis_x in range(dis_w):

                #几何中心相同
                sou_x = (dis_x + 0.5)*scale_x-0.5
                sou_y = (dis_y + 0.5)*scale_y-0.5

                #找到用于插值计算的坐标点
                sou_x0 = int(np.floor(sou_x))
                sou_x1 = min(sou_x0 + 1, sou_w - 1)
                sou_y0 = int(np.floor(sou_y))
                sou_y1 = min(sou_y0 + 1, sou_h - 1)

                #插值计算
                temp0 = (sou_x1 - sou_x) * img[sou_y0, sou_x0, i] + (sou_x - sou_x0) * img[sou_y0, sou_x1, i]
                temp1 = (sou_x1 - sou_x) * img[sou_y1, sou_x0, i] + (sou_x - sou_x0) * img[sou_y1, sou_x1, i]
                dis_img[dis_y, dis_x, i] = int((sou_y1 - sou_y) * temp0 + (sou_y - sou_y0) * temp1)

    return dis_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dis = bilinear_interpolation(img, (1000, 1000))
    cv2.imshow('bilinear interp', dis)
    cv2.imshow('img', img)
    cv2.waitKey(0)

