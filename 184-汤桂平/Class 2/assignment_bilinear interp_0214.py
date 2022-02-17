import numpy as np
import cv2

def BilinearInterp(img, dst_dim):
    dst_h = dst_dim[0]
    dst_w = dst_dim[1]
    src_h, src_w, channels = img.shape
    if dst_h == src_h and dst_w == src_w:
        return img
    DstImage = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)
    for i in range(channels):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                src_x = (dst_x + 0.5) * src_h / dst_h - 0.5
                src_y = (dst_y + 0.5) * src_w / dst_w - 0.5
                src_x0 = int(np.floor(src_x))    # np.floor()函数是向下取整，但返回值是浮点型
                src_x1 = min(src_x0 + 1, src_h - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_w - 1)
                temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]
                temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]
                DstImage[dst_x, dst_y, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return DstImage

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    zoom_img = BilinearInterp(img, (800,800))
    cv2.imshow('zoom_img', zoom_img)
    cv2.imshow('src_img', img)
    cv2.waitKey(0)



