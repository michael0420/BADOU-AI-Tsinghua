import cv2
import numpy as np

# 读取图片
img = cv2.imread('photo1.jpg')

result3 = img.copy()  # 拷贝一副图像副本

'''
确定原图和目标图像的顶点坐标，一共四组
'''
src_QuadVer = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst_QuadVer = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

print(img.shape) # 打印出图像的边界

'''
根据原图和目标图形的顶点坐标，得到透视变换矩阵，使用函数：
    getPerspectiveTransform(src, dst, solveMethod=None)
    src：原图像顶点
    dst：目标图像顶点
'''
m = cv2.getPerspectiveTransform(src_QuadVer, dst_QuadVer)
print("WarpMatrix:\n", m) # 打印出得到的透视变换矩阵

'''
使用得到的透视变换矩阵进行变换,
    warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
    src：需要进行透视变换的原图像
    M：透视变换矩阵
    dsize：输出图像大小
    dst：目标图像
'''
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src img", img)
cv2.imshow("result", result)
cv2.waitKey(0)

