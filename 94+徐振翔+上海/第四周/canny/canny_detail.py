# 目标：手写实现canny的详细算法，
# 双阈值及soble核大小实现进度条可控，
# 添加接口调用canny官方接口或自制接口
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

'''
canny 接口选择，1：自制接口; 2：官方接口; 3：不执行三滑块部分调用
滑块窗口使用esc退出，退出时返回当前canny图像
# 注意：使用自制接口时，三滑块窗口会很卡，因为自制接口算的慢
'''
canny_type = 2


def cv_show(im, name="demo"):
    import cv2
    import numpy as np
    if (type(name) != str) or (type(im) != np.ndarray):
        return
    # show image
    cv2.imshow(name, im)
    # wait a key to destroy window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# im = cv2.imread(".\..\..\general_lib\lenna.png")
# cv_show(im)


def canny_detail(gray_pic, lowth, highth, gs_size=3, sobel_size=3):
    '''
    边缘提取5个步骤：
    1、灰度化
    2、滤波，本函数选择高斯函数
    3、用算子做卷积，本函数选择sobel
    4、对梯度进行非极大值抑制
    5、双阈值检测及边缘链接
    步骤 1 通过opencv自带的灰度化函数实现,即当前函数接口输入图像仅支持灰度图
    输入参数：
    gray_pic：灰度图
    lowth：阈值下限
    highth：阈值上线
    gs_size:高斯核大小，目前仅支持3\5\7
    sobel_size:sobel算子的大小，目前仅支持3与5
    '''
    # 简易参数检查
    if (not (gray_pic.ndim == 2)) or (lowth < 0) or (highth < 1) or (highth < lowth) or \
            (not ((gs_size == 3) or (gs_size == 5) or (gs_size == 7))) or (
    not ((sobel_size == 3) or (sobel_size == 5))):
        print("err")
        return []
    img = np.array(gray_pic)

    # 2、滤波，本函数选择高斯函数(高斯平滑)
    # sigma = 1.4  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = gs_size  # 存储计划用的高斯核函数的尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列,作为高斯核的x轴坐标
    # print("tmp:\n", tmp)
    # 高斯分布： 1/((sqrt(2pi)^(维度))*a) * e^(-(x-u)^2/(2a^2))
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核，1/(2pi*a^2)
    n2 = -1 / (2 * sigma ** 2)  # -1/(2*a^2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 归一化
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # print("gs_kernel\n", Gaussian_filter)
    # 高斯滤波
    dx, dy = img.shape
    img_gs = np.zeros(img.shape)  # 存储高斯平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    # array——表示需要填充的数组；
    # pad_width——表示每个轴（axis）边缘需要填充的数值数目。
    # 参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)
    # 表示第1轴两边缘分别填充before_1个和after_1个数值。取值为：{sequence, array_like, int}
    # mode——表示填充的方式（取值：str字符串或用户提供的函数）, 总共有11种填充模式；
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 填补pad
    # cv_show(img_pad)
    # 高斯核卷积
    for i in range(dx):
        for j in range(dy):
            img_gs[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    # 必须转成u8，否则会出现全白图像
    # # cv_show(img_gs.astype(np.uint8))
    # plt.figure(1)
    # plt.imshow(img_gs.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    # plt.axis('off')

    # 3、使用sobel算子求梯度，以下是支持的不同池逊的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_3_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_3_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_kernel_5_x = np.array([[-1, -2, 0, 2, 1],
                                 [-4, -8, 0, 8, 4],
                                 [-6, -12, 0, 12, 6],
                                 [-4, -8, 0, 8, 4],
                                 [-1, -2, 0, 2, 1]])
    sobel_kernel_5_y = np.array([[-1, -4, -6, -4, -1],
                                 [-2, -8, -12, -8, -2],
                                 [0, 0, 0, 0, 0],
                                 [2, 8, 12, 8, 2],
                                 [1, 4, 6, 4, 1]])
    if sobel_size == 3:
        sobel_kernel_x = sobel_kernel_3_x.view()
        sobel_kernel_y = sobel_kernel_3_y.view()
    else:
        sobel_kernel_x = sobel_kernel_5_x.view()
        sobel_kernel_y = sobel_kernel_5_y.view()

    img_tidu_x = np.zeros(img_gs.shape)  # 存储梯度图像
    img_tidu_y = np.zeros((dx, dy))  # y方向梯度，输入参数可以由多种方法输入,建议用()
    img_tidu = np.zeros([dx, dy])  # 合并后的梯度图
    tmp = sobel_size // 2
    img_pad = np.pad(img_gs, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补，根据sobel核矩阵结构填充

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + sobel_size, j:j + sobel_size] * sobel_kernel_x)  # x方向梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + sobel_size, j:j + sobel_size] * sobel_kernel_y)  # y方向梯度
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 整体梯度
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 防止除数为0
    angle = img_tidu_y / img_tidu_x  # 梯度方向（角度）
    # # cv_show(img_tidu.astype(np.uint8))
    # plt.figure(2)
    # plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    # 4、对梯度进行非极大值抑制
    img_yz = np.zeros_like(img_tidu)
    # 边界可选是否处理
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            tmp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 获取八邻域矩阵，注意[]左开右闭
            t = True
            if angle[i, j] >= 1:
                t1 = (tmp[0, 2] - tmp[0, 1]) / angle[i, j] + tmp[0, 1]
                t2 = (tmp[2, 0] - tmp[2, 1]) / angle[i, j] + tmp[2, 1]
                if img_tidu[i, j] < t1 or img_tidu[i, j] < t2:
                    t = False
            elif angle[i, j] <= -1:
                t1 = (tmp[0, 1] - tmp[0, 0]) / angle[i, j] + tmp[0, 1]
                t2 = (tmp[2, 1] - tmp[2, 2]) / angle[i, j] + tmp[2, 1]
                if img_tidu[i, j] < t1 or img_tidu[i, j] < t2:
                    t = False
            elif angle[i, j] > 0:
                t1 = (tmp[0, 2] - tmp[1, 2]) * angle[i, j] + tmp[1, 2]
                t2 = (tmp[2, 0] - tmp[1, 0]) * angle[i, j] + tmp[1, 0]
                if img_tidu[i, j] < t1 or img_tidu[i, j] < t2:
                    t = False
            elif angle[i, j] < 0:
                t1 = (tmp[1, 0] - tmp[0, 0]) * angle[i, j] + tmp[1, 0]
                t2 = (tmp[1, 2] - tmp[2, 2]) * angle[i, j] + tmp[1, 2]
                if img_tidu[i, j] < t1 or img_tidu[i, j] < t2:
                    t = False
            if t:
                img_yz[i, j] = img_tidu[i, j]

    # # cv_show(img_yz)
    # plt.figure(3)
    # plt.imshow(img_yz.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    # 5、双阈值检测及边缘链接
    # 双阈值检测，连接边缘。遍历所有一定是边的点, 查看8邻域是否存在有可能是边的点，进栈
    # 最终结果：强边缘为边缘，存在链接到强边缘的弱边缘也为边缘
    zhan = []
    img_jc = img_yz.copy()
    # 可不考虑外圈
    for i in range(1, img_yz.shape[0] - 1):
        for j in range(1, img_yz.shape[1] - 1):
            if img_jc[i, j] >= highth:  # 强边缘
                img_jc[i, j] = 255  # 标记为边缘
                zhan.append([i, j])
            elif img_jc[i, j] <= lowth:  # 非边缘
                img_jc[i, j] = 0  # 标记为非边缘

    # 每次加入新确认的边缘，去确认他的周围是否有边缘，直到没有新边缘加入为止，用堆栈处理
    while not len(zhan) == 0:
        x, y = zhan.pop()
        a = img_jc[x - 1:x + 2, y - 1:y + 2]
        for i in range(3):
            for j in range(3):
                if lowth < a[i, j] < highth:
                    img_jc[x - 1 + i, y - 1 + j] = 255  # 标记为边缘
                    zhan.append([x - 1 + i, y - 1 + j])

    for i in range(img_jc.shape[0]):
        for j in range(img_jc.shape[1]):
            if img_jc[i, j] != 0 and img_jc[i, j] != 255:
                img_jc[i, j] = 0

    return img_jc.astype(np.uint8)
    # # cv_show(img_jc)
    # 绘图
    # plt.figure(4)
    # plt.imshow(img_jc.astype(np.uint8), cmap='gray')
    # plt.axis('off')  # 关闭坐标刻度值
    # plt.show()


def nothing():
    pass


# def CannyThreshold(lowThreshold, highThreshold, kernel_size):
def CannyThreshold():
    global detected_edges
    lowThreshold = cv2.getTrackbarPos('Min_threshold', windown_name)
    highThreshold = cv2.getTrackbarPos('Max_threshold', windown_name)
    kernel_size = cv2.getTrackbarPos('kernel_size', windown_name)
    if highThreshold < lowThreshold:
        highThreshold = lowThreshold
    kernel_size = int(kernel_size / 2) * 2 + 1
    if kernel_size < 3:
        kernel_size = 3
    print(lowThreshold, highThreshold, kernel_size)
    # detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    # detected_edges = cv2.Canny(detected_edges,
    #                            lowThreshold,
    #                            highThreshold,
    #                            apertureSize=kernel_size)  # 边缘检测
    if canny_type == 2:
        detected_edges = cv2.Canny(gray, lowThreshold, highThreshold,
                                   apertureSize=kernel_size)  # 边缘检测
    elif canny_type == 1:
        detected_edges = canny_detail(gray, lowThreshold, highThreshold, sobel_size=kernel_size)
        # detected_edges = img_jc(gray,
        #                            lowThreshold,
        #                            highThreshold,
        #                            apertureSize=kernel_size)  # 边缘检测
        # pass
    else:
        sys.exit()

    # just add some colours to edges from original image.
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上
    cv2.imshow(windown_name, dst)


if __name__ == "__main__":
    '''
    三进度条实现双阈值及核函数大小选型
    '''
    img = cv2.imread('lenna.png')  # 读入灰度图
    # cv_show(img)
    # 多种方法获取灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

    # canny_detail(gray, 10, 200)
    # canny_detail(gray, 10, 200, gs_size=5)

    pass

    lowThreshold = 0
    highThreshold = 1
    kernel_size = 3
    windown_name = 'canny demo'
    img = cv2.imread('lenna.png')  # 读入灰度图
    # cv_show(img)
    # 多种方法获取灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
    # gray = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow(windown_name)

    # 设置调节杠,多个滑块需要特殊处理
    '''
    下面是第二个函数，cv2.createTrackbar()
    共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
    第一个参数，是这个trackbar对象的名字
    第二个参数，是这个trackbar对象所在面板的名字
    第三个参数，是这个trackbar的默认值,也是调节的对象
    第四个参数，是这个trackbar上调节的范围(0~count)
    第五个参数，是调节trackbar时调用的回调函数名
    '''
    cv2.createTrackbar('Min_threshold', windown_name, 0, 800, nothing)
    cv2.createTrackbar('Max_threshold', windown_name, 1, 800, nothing)
    cv2.createTrackbar('kernel_size', windown_name, 3, 5, nothing)

    detected_edges = []

    detected_edges = cv2.Canny(gray, lowThreshold, highThreshold, apertureSize=kernel_size)  # 边缘检测
    print(type(detected_edges))
    print(detected_edges)
    print('-' * 40)

    detected_edges = canny_detail(gray, lowThreshold, highThreshold, sobel_size=kernel_size)
    print(type(detected_edges))
    print(detected_edges)

    CannyThreshold()  # initialization

    # CannyThreshold(lowThreshold, highThreshold, kernel_size)  # initialization
    while 1:
        # lowThreshold = cv2.getTrackbarPos('Min_threshold', windown_name)
        # highThreshold = cv2.getTrackbarPos('Max_threshold', windown_name)
        # kernel_size = cv2.getTrackbarPos('kernel_size', windown_name)
        # CannyThreshold(lowThreshold,highThreshold,kernel_size)
        CannyThreshold()
        if cv2.waitKey(1) == 27:  # wait for ESC key to exit cv2
            cv2.destroyAllWindows()
            # sys.wait(1)
            cv_show(detected_edges)
            break
