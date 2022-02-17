import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import cv2
import imutils
from PIL import Image

mpl.rcParams['font.sans-serif'] = 'KaiTi'
pic = 'lenna.png'
img = cv2.imread(pic, 0)  # 以灰度模式读取照片
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title("原图")
plt.imshow(imutils.opencv2matplotlib(img))
print(img)

# 1.高斯平滑
sigma = 0.1


def GUSSFL(sigma):
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:  # 最好是奇数，不是奇数则加1
        dim += 1
    Gsfl = np.zeros([dim, dim])  # 是数组不是列表
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gsfl[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gsfl = Gsfl / Gsfl.sum()  # 归一化
    return Gsfl, dim


Gsfl, dim = GUSSFL(1.52)
dx, dy = img.shape
img_new = np.zeros(img.shape, np.uint8)
tmp = dim // 2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
# np.pad(array,((x1,x2),(y1,y2)),mode) x1为数据前填充行数，x2为数据后填充行数，y（表列数）也一样
print(img_pad)
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gsfl)
plt.subplot(2, 2, 2)
plt.axis('off')
plt.title("高斯平滑图")
plt.imshow(imutils.opencv2matplotlib(img_new))
print(img_new)

# --------------------------------------------------------------------------- #

# 滑条实现sigma与图像模糊成都的关系
img_new1 = np.zeros(img.shape, np.uint8)


def nothing(x):
    pass


cv2.namedWindow('GSblur')
cv2.createTrackbar('sigma', 'GSblur', 5, 300, nothing)
while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    cv2.imshow('GSblur', img_new1)
    sigma = cv2.getTrackbarPos('sigma', 'GSblur') / 100
    Gsfl, dim = GUSSFL(sigma)
    dx, dy = img.shape
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new1[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gsfl)
cv2.destroyAllWindows()

# --------------------------------------------------------------------------- #

# 2.求梯度,利用的是局部加强的sobel算子
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 检测竖直边缘
sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 检测水平边缘
Tx = np.zeros(img_new.shape)   # 存储梯度图像
Ty = np.zeros(img_new.shape)
Ts = np.zeros(img_new.shape)
img_pad=np.pad(img_new,((1,1),(1,1)),'constant') # peddle 边缘填补
for i in range(dx):
    for j in range(dy):
        Tx[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sx) # x方向，竖直边缘
        Ty[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sy) # y方向，水平边缘
        Ts[i,j]=np.sqrt(Tx[i,j]**2+Ty[i,j]**2)
Tx[Tx==0]=0.00000001
angle=Ty/Tx
plt.subplot(2,2,3)
plt.title("梯度图")
plt.imshow(Ts.astype(np.uint8),cmap='gray')
plt.axis('off')

# 3.非极大值抑制
img_yz=np.zeros(Ts.shape)
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = Ts[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (Ts[i, j] > num_1 and Ts[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (Ts[i, j] > num_1 and Ts[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (Ts[i, j] > num_1 and Ts[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (Ts[i, j] > num_1 and Ts[i, j] > num_2):
                flag = False
        if flag:
            img_yz[i, j] = Ts[i, j]
plt.subplot(2,2,4)
plt.axis('off')
plt.title('非极大值抑制')
plt.imshow(img_yz.astype(np.uint8),cmap='gray')
plt.show()

# 4.双阈值检测
lb=Ts.mean()*0.5  # 低阈值
hb=lb*3  # 高阈值
zhan=[]   #用来检测8邻点是否为强边缘点
for i in range(1,img_yz.shape[0]-1):   # 不考虑外圈
    for j in range(1,img_yz.shape[1]-1):
        if img_yz[i,j]>=hb:
            img_yz[i,j]=255
            zhan.append([i,j])
        elif img_yz[i,j]<=lb:  # 舍
            img_yz[i,j]=0  # 未标记的默认为弱边缘点
while not len(zhan) ==0:
    tp1,tp2=zhan.pop() # 出栈   tp1为横坐标，tp2为纵坐标
    a=img_yz[tp1-1:tp1+2,tp2-1:tp2+2]
    # 检测强边缘点的8临点是否有弱边缘点，若有，则将其变为强边缘点，并进栈
    if (a[0, 0] < hb) and (a[0, 0] > lb):
        img_yz[tp1 - 1, tp2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([tp1 - 1, tp2 - 1])  # 进栈
    if (a[0, 1] < hb) and (a[0, 1] > lb):
        img_yz[tp1 - 1, tp2] = 255
        zhan.append([tp1 - 1, tp2])
    if (a[0, 2] < hb) and (a[0, 2] > lb):
        img_yz[tp1 - 1, tp2 + 1] = 255
        zhan.append([tp1 - 1, tp2 + 1])
    if (a[1, 0] < hb) and (a[1, 0] > lb):
        img_yz[tp1, tp2 - 1] = 255
        zhan.append([tp1, tp2 - 1])
    if (a[1, 2] < hb) and (a[1, 2] > lb):
        img_yz[tp1, tp2 + 1] = 255
        zhan.append([tp1, tp2 + 1])
    if (a[2, 0] < hb) and (a[2, 0] > lb):
        img_yz[tp1 + 1, tp2 - 1] = 255
        zhan.append([tp1 + 1, tp2 - 1])
    if (a[2, 1] < hb) and (a[2, 1] > lb):
        img_yz[tp1 + 1, tp2] = 255
        zhan.append([tp1 + 1, tp2])
    if (a[2, 2] < hb) and (a[2, 2] > lb):
        img_yz[tp1 + 1, tp2 + 1] = 255
        zhan.append([tp1 + 1, tp2 + 1])

# 二值化，将没检测到的弱边缘抑制
for i in range(img_yz.shape[0]):
    for j in range(img_yz.shape[1]):
        if img_yz[i, j] != 0 and img_yz[i, j] != 255:
            img_yz[i, j] = 0

# 绘图
plt.imshow(img_yz.astype(np.uint8),cmap='gray')
plt.title("双阈值检测")
plt.axis('off')
plt.show()

print('-------------接口实现--------------')
def CannyThreshold(lb):
    dg=cv2.GaussianBlur(gray,(7,7),0)  # g
    # GuassianBlur(img,(a,b),0) a,b为高斯核长宽，标准差为0
    dg=cv2.Canny(dg,lb,lb*3,apertureSize=Kernel_size) #边缘检测

    dst = cv2.bitwise_and(img1, img1, mask=dg)  # 用原始颜色添加到检测的边缘上
    cv2.imshow('canny demo', dst)
lb=0
maxlb=100
Kernel_size=3
img1=cv2.imread('lenna.png')
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')
#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('Min threshold', 'canny demo', lb, maxlb, CannyThreshold)
CannyThreshold(0)   # initialization
if cv2.waitKey(0)==27:   # 等待Esc相应
    cv2.destroyAllWindows()