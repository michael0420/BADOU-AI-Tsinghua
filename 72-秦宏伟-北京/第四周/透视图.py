import cv2
import numpy as np

def matrix(A,B):
    matix = np.zeros([8,8])
    exp = np.zeros([8, 1])
#   验证A、B的有效性，暂时忽略
    for i in range(4):
        matix[2*i, 0] = A[i][0] #X0
        matix[2*i, 1] = A[i][1] #Y0
        matix[2*i, 2] = 1
        matix[2*i, 6] = -A[i][0]*B[i][0] #-X0*X'0
        matix[2*i, 7] = -A[i][1]*B[i][0] #-Y0*X'0
        exp[2*i] = B[i][0]
        matix[2*i+1, 3] = A[i][0]
        matix[2*i+1, 4] = A[i][1]
        matix[2*i+1, 5] = 1
        matix[2*i+1, 6] = -A[i][0] * B[i][1]  # -X0*Y'0
        matix[2*i+1, 7] = -A[i][1] * B[i][1]  # -Y0*Y'0
        exp[2 * i +1] = B[i][1]

    matix = np.mat(matix)
    wrapMatrix = matix.I * exp
    wrapMatrix = np.array(wrapMatrix)
    wrapMatrix = np.append(wrapMatrix,1)
    wrapMatrix = np.reshape(wrapMatrix,[3,3])

    return wrapMatrix


img = cv2.imread('photo1.jpg')
result3 = img.copy()
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
# m = cv2.getPerspectiveTransform(src, dst)

#计算透视矩阵
m = matrix(src,dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
