import numpy as np
import cv2


# 最邻近插值
def nearest_interpolation(image, dst_h, dst_w):
    src_h, src_w, channels = image.shape  # 获取原图尺寸
    zoom_mut_h = dst_h/src_h  # 高的缩放比例
    zoom_mut_w = dst_w/src_w  # 宽的缩放比例
    nearest_int_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            src_x = int(i/zoom_mut_w+0.5)  # int向下取整，采用+0.5的方式进行四舍五入
            src_y = int(j/zoom_mut_h+0.5)
            nearest_int_img[i, j] = image[src_x, src_y]  # 像素赋值
    return nearest_int_img


if __name__ == '__main__':
    image = cv2.imread("lenna.png")  # 读取图像
    cv2.imshow("lenna", image)  # 显示原图图像
    dst_h, dst_w = (800, 800)  # 插值后图像尺寸
    nearest_int = nearest_interpolation(image, dst_h, dst_w)  # 最邻近插值
    cv2.imshow("nearest_interpolation_image", nearest_int)  # 显示插值后的图像
    cv2.imwrite("nearest_interpolation_image.png", nearest_int)  # 保存插值后的图像
    cv2.waitKey()  # 等待操作
    cv2.destroyAllWindows()  # 关闭显示图像的窗口
