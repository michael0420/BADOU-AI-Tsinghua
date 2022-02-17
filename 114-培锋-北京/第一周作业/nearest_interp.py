'''
最近邻插值(放大、缩小图片)
2021/11/29--2021/11/30
原  图：src_x,src_y    src_w,src_h
现  图：dst_x,dst_y    dst_w,dst_h

公式：src_x / dst_x = src_w / dst_w
      src_y / dst_y = src_h / dst_h
==>
    dst_x = src_x / (src_w / dst_w)
    dst_y = src_y / (src_h / dst_h)

    src_x = (src_w / dst_w) * dst_x
    src_y = (src_h / dst_h) * dst_y
==>
    g( dst_x , dst_y ) = f{ (src_w / dst_w) * dst_x , (src_h / dst_h) * dst_y}

'''
import cv2
import numpy

def function(src_imag,pix_x,pix_y): #src_imagde:原图像;目标图像尺寸:pix_x * pix_y
    src_w,src_h,src_chanel = src_imag.shape
    dst_imag = numpy.zeros((pix_x,pix_y,src_chanel),numpy.uint8)
    for i in range(pix_x):
        for j in range(pix_y):
            dst_imag[i, j] = src_imag[int(i * (src_w / pix_x)), int(j * (src_h / pix_y))]
    return dst_imag

imag = cv2.imread("F:/cycle_gril/lenna.png")
cv2.imshow("src image",imag)

#放大
dst = function(imag,800,800)
cv2.imshow("out image",dst)

#缩小
dst1 = function(imag,200,200)
cv2.imshow("out image1",dst1)

#缩小
dst2 = function(imag,100,100)
cv2.imshow("out image2",dst2)
cv2.waitKey(0)

