'''
双线性插值
2021/11/30--2021/12/01
对应关系（来自最邻近插值推导）:
    src_x  = (src_w / dst_w) * dst_x
    src_y  = (src_h / dst_h) * dst_y
使输出图像和输入图像几何中心重合后推导出以下公式：
    src_x + 0.5 = (src_w / dst_w) * (dst_x + 0.5)
    src_y + 0.5 = (src_h / dst_h) * (dst_y + 0.5)
==>
    src_x = (src_w / dst_w) * (dst_x + 0.5) - 0.5
    src_y = (src_h / dst_h) * (dst_y + 0.5) - 0.5
'''

'''
            |                           |                |
 -y2-----Q12(src_x1,src_y2)-----------R2-----------Q22(src_x2,src_y2)--------
            |                           |                |
            |                           |                |
 -y---------|--------------------p(src_x,src_y)--------------------------
            |                           |                |
            |                           |                |
            |                           |                |
 -y1-----Q11(src_x1,src_y1)------------R1-----------Q22(src_x2,src_y1)--------
            |                           |                |
            |                           |                |
            x1                          x               x2
 '''
#三次插值之后，推导得出公式：f(x,y)=(y2-y)( (x2-x)f(Q11)+(x-x1)f(Q21) ) + (y-y1)( (x2-x)f(Q12)+(X-X1)f(Q22) )
# 获取目标像素点p(x,y)临近四个像素点坐标,也就是Q11,Q12,Q21,Q22
# Q12(x1,y2)        Q22(x2,y2)
# Q11(x1,y1)        Q22(x2,y1)
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy
import cv2

def function(src_image,pix_x,pix_y):
    src_w,src_h,src_channel = src_image.shape
    dst_w = pix_x
    dst_h = pix_y
    if src_w == dst_w and src_h == dst_h:
        return src_image.copy()

    dst_image = numpy.zeros((dst_w ,dst_h,src_channel),dtype = numpy.uint8)
    for dst_x in range(dst_w):
        for dst_y in range(dst_h):
            for i in range(src_channel):
                #1.找像素位置(中心对称)
                src_x  = (float(src_w / dst_w)) * (dst_x + 0.5) - 0.5
                src_y  = (float(src_h / dst_h)) * (dst_y + 0.5) - 0.5
                #2.获取目标图像中某一个像素临近的四个像素坐标。可对应到ppt中的Q11,Q12,Q21,Q22
                src_x1 = int(numpy.floor(src_x))#向下取整
                src_y1 = int(numpy.floor(src_y))#向下取整
                src_x2 = min(src_x1 + 1, src_w - 1)#若加1之后超过图像边界，就取图像边界值
                src_y2 = min(src_y1 + 1, src_h - 1)
                #3.代入公式:f(x,y)=(y2-y)( (x2-x)f(Q11)+(x-x1)f(Q21) ) + (y-y1)( (x2-x)f(Q12)+(X-X1)f(Q22) )
                #           f(x,y)=(y2-y)(        a    +     b        ) + (y-y1)(       c    +      d      )
                a = (src_x2-src_x) * src_image[src_x1,src_y1,i]
                b = (src_x-src_x1) * src_image[src_x2,src_y1,i]
                c = (src_x2-src_x) * src_image[src_x1,src_y2,i]
                d = (src_x-src_x1) * src_image[src_x2,src_y2,i]
                dst_image[dst_x, dst_y, i] = int((src_y2 - src_y) * (a+b) + (src_y - src_y1) * (c+d))
    return dst_image

src_image = cv2.imread("F:/cycle_gril/lenna.png")
cv2.imshow("input mage",src_image)

dst_image = function(src_image,900,900)
cv2.imshow("out image",dst_image)
cv2.waitKey(0)

