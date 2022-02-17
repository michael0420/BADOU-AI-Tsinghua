import numpy as np
import cv2


def bilinear_interpolation(image, dst_h, dst_w):
    src_h, src_w, channels = image.shape  # 获取原图尺寸
    zoom_mut_h = dst_h / src_h  # 高的缩放比例
    zoom_mut_w = dst_w / src_w  # 宽的缩放比例
    bilinear_int_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    for n in range(3):
        for i in range(dst_h):
            for j in range(dst_w):
                src_x = (i + 0.5) / zoom_mut_w - 0.5  # 像素坐标对应关系
                src_y = (j + 0.5) / zoom_mut_h - 0.5
                src_x0 = max(int(np.floor(src_x)), 0)  # 相邻像素点坐标
                src_y0 = max(int(np.floor(src_y)), 0)
                src_x1 = min(int(np.ceil(src_x)), src_w-1)
                src_y1 = min(int(np.ceil(src_y)), src_h-1)
                bilinear_int_img[i, j, n] = int((src_y1-src_y)*((src_x1-src_x)*image[src_x0, src_y0, n]+(src_x-src_x0)*image[src_x1, src_y0, n]) + (src_y-src_y0)*((src_x1-src_x)*image[src_x0, src_y1, n]+(src_x-src_x0)*image[src_x1, src_y1, n]))  # 像素赋值
    return bilinear_int_img


if __name__ == '__main__':
    image = cv2.imread("lenna.png")  # 读取图像
    cv2.imshow("lenna", image)  # 显示原图图像
    dst_h, dst_w = (800, 800)  # 插值后图像尺寸
    bilinear_int = bilinear_interpolation(image, dst_h, dst_w)  # 双线性近插值
    cv2.imshow("bilinear_interpolation_image", bilinear_int)  # 显示插值后的图像
    cv2.imwrite("bilinear_interpolation_image.png", bilinear_int)  # 保存插值后的图像
    cv2.waitKey()  # 等待操作
    cv2.destroyAllWindows()  # 关闭显示图像的窗口
