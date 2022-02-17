import cv2 
import numpy as np

flag =True
if flag:
    img = cv2.imread("lenna.png")
    src_h, src_w, channel = img.shape

    dst_img = np.zeros((700, 700, 3), dtype=np.uint8)
    dst_h, dst_w = dst_img.shape[:2]
    if src_h == dst_h and src_w == dst_w:
        print("和原图一样")
    # scale_x  x方向的比例
    # scale_y  y 方向的比例

    scale_x, scale_y = float(src_w / dst_w), float(src_h / dst_h)
    for i in range(3):  # 3 通道
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 目标图像和原图像的中心点对齐
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 带入计算公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    print(dst_h, dst_w)
    print(src_h, src_w)

    cv2.imshow("img", img)
    cv2.imshow("dst", dst_img)
    cv2.waitKey(0)

#flag = True
flag =False
if flag :
    img = cv2.imread("lenna.png") # 读入一张图片
    bil_inter= cv2.resize(img,dsize=None,fx=1.2,fy=1.2,interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img",img)
    cv2.imshow("bil_inter",bil_inter)
    cv2.waitKey(0)
