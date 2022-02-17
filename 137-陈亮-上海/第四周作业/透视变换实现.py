import cv2
import numpy as np

# 1,寻找顶点.2,制作warpMatrix.3,进行透视变换,将对应坐标计算得出,无需手动输入
img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
result3 = img.copy()
def findPoint(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    dilate = cv2.dilate(blurred,np.ones((3,3)),iterations=1)
    edage = cv2.Canny(dilate,50,100)
    # 寻找轮廓
    cntPoints = None
    cnts = cv2.findContours(edage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(edage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0]
    if len(cnts) > 0:
        # 对寻找到的轮廓排序
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            length = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.02*length,True)
            if len(approx) == 4:
                cntPoints = approx
            else:
                print('无法找到对应四个点')
                break
            return cntPoints

def warpMatrix(srcPoints,dstPoints):
    assert srcPoints.shape[0] == dstPoints.shape[0] and srcPoints.shape[0] >= 4
    nums = srcPoints.shape[0]
    A = np.zeros((2*nums,8))   #A(8,8)*warpMatrix(8,1) = B(8,1)
    B = np.zeros((2*nums,1))
    for i in range(0,nums):
        #将图片矩阵,待处理矩阵读入A_i,B_i
        A_i = srcPoints[i,:]
        B_i = dstPoints[i,:]
        A[i*2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[i*2] = B_i[0]
        A[i*2+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[i*2+1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

src_list = []
# 获取顶点坐标，作为输入值
for peak in findPoint(gray):
    peak = peak.tolist()
    src_list.append(peak[0])
src = src_list
src = np.array(src)

# 疑问：src可以以逆时针方向读出，对应到dst图片以[0,0]开始却是顺时针方向
# 由顶点值获取变换后的长宽坐标
h1 = np.sqrt((pow((src[0][0] - src[1][0]),2)+pow((src[1][1] - src[0][1]),2)))
h2 = np.sqrt((pow((src[3][0] - src[2][0]),2)+pow((src[2][1] - src[3][1]),2)))
h = int((h1+h2)/2)
w1 = np.sqrt((pow((src[3][0] - src[0][0]),2)+pow((src[3][1] - src[0][1]),2)))
w2 = np.sqrt((pow((src[2][0] - src[1][0]),2)+pow((src[2][1] - src[1][1]),2)))
w = int((w1+w2)/2)
dst = [[0,0], [0,h], [w,h], [w,0]]
dst = np.array(dst)

m = warpMatrix(src,dst)
result = cv2.warpPerspective(result3, m, (w,h))
cv2.namedWindow('src',cv2.WINDOW_NORMAL)
cv2.namedWindow('result',cv2.WINDOW_NORMAL)
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)