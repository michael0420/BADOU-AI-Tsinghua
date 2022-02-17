import cv2
import numpy as np


def billiner(src_img, dst_shape):
    src_h, src_w, channel = src_img.shape
    dst_h, dst_w = dst_shape[0], dst_shape[1]

    dst_out_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_y, scale_x = src_h / dst_h, src_w / dst_w
    print("src_img_dtype=%s" % src_img.dtype)
    print('src_h=%s,src_w=%s,dst_h=%s,dst_w=%s ,channel=%s' % (src_h, src_w, dst_w, dst_w, channel))
    print('scale_x=%s,scale_y=%s' % (scale_x, scale_y))
    for channel_index in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 求应该取原图的像素值索引
                y, x = (dst_y + 0.5) * scale_y - 0.5, (dst_x + 0.5) * scale_x - 0.5

                y1 = int(np.floor(y))
                x1 = int(np.floor(x))

                y2 = min(y1 + 1, src_h - 1)
                x2 = min(x1 + 1, src_w - 1)
                print('y1=%s,y2=%s,x1=%s,x2=%s,x=%s,y=%s' % (y1, y2, x1, x2, x, y))
                # 求x方向的值
                r1 = (x2 - x) * src_img[y1, x1, channel_index] + (x - x1) * src_img[y2, x1, channel_index]
                r2 = (x2 - x) * src_img[y1, x2, channel_index] + (x - x1) * src_img[y2, x2, channel_index]
                # 求y方向的值
                p = int((y2 - y) * r1 + (y - y1) * r2)

                dst_out_img[dst_y, dst_x, channel_index] = p

    return dst_out_img


if __name__ == '__main__':
    src_img = cv2.imread('lenna.png')
    dst_shape = (800, 800)
    dst_out_img = billiner(src_img, dst_shape)

    cv2.imshow('billiner', dst_out_img)
    cv2.waitKey()
