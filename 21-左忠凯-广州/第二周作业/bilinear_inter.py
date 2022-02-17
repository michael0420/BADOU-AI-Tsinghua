import numpy as np
import cv2

def bilinear_inter(dst_h, dst_w, src_img):
    src_h = src_img.shape[0]
    src_w = src_img.shape[1]
    channel = src_img.shape[2]

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    #算出目标图像和原始图像的比例
    x_scale = float(src_w) / dst_w
    y_scale = float(src_h) / dst_h

    print("src_h = {}, src_w = {}".format(src_h, src_w))
    print("dst_h = {}, dst_w = {}".format(dst_h, dst_w))

    for i in range(3): # 处理RGB三个通道
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                # 找到原图中对应的像素点，变换后的图像和原图的几何中心要对齐
                src_x = (dst_x + 0.5) * x_scale - 0.5
                src_y = (dst_y + 0.5) * y_scale - 0.5

                # 算出x0,x1,y0,y1
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1) # 防止超出图像边界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1) # 防止超出图像边界

                temp0 = (src_x1 - src_x) * src_img[src_x0, src_y0, i] + (src_x - src_x0) * src_img[src_x1, src_y0, i]
                temp1 = (src_x1 - src_x) * src_img[src_x0, src_y1, i] + (src_x - src_x0) * src_img[src_x1, src_y1, i]
                dst_img[dst_x, dst_y, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst_img = bilinear_inter(700, 700, img)
    cv2.imshow("source img", img)
    cv2.imshow("bilinear img", dst_img)
    cv2.waitKey(0)

