import numpy as np
import matplotlib.pyplot as plt #绘图
import math

#canny算法：https://www.cnblogs.com/techyan1990/p/7291771.html

'''
边缘检测算法：
sobel
prewitt
laplace
canny
'''
if __name__ == '__main__':
    pic_path = 'F:/cycle_gril/lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值就是灰度化了

    #1.高斯平滑处理
    #设定高斯核大小，要为奇数
    dim = 5

    Gaussian_filter = np.zeros([dim,dim])#建立矩阵，存储高斯核的值
    #Gaussian_filter = [[]] #学习阶段，也可以人为设定一个高斯核

    #1.1高斯函数中的一些参数
    sigma = 2
    a = 1/(2*math.pi*sigma**2) #高斯函数最左边的系数
    b = -1/(2*sigma**2) #高斯函数指数的系数
    #1.2根据以下坐标计算高斯核
    #       (-2,2)    (-1,2)   (0,2)   (1,2)   (2,2)
    #       (-2,1)    (-1,1)   (0,1)   (1,1)   (2,1)
    #       (-2,0)    (-1,0)   (0,0)   (1,0)   (2,0)
    #       (-2,-1)   (-1,-1)  (0,-1)  (1,-1)  (2,-1)
    #       (-2,-2)   (-1,-2)  (0,-2)  (1,-2)  (2,-2)

    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i][j] = a * math.exp(b*((i-dim//2)**2 + (j-dim//2)**2))

    print(Gaussian_filter)
    print(Gaussian_filter.sum())
    #1.3归一化处理
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    print(Gaussian_filter)

    #1.4高斯滤波
    dst_x,dst_y = img.shape
    img_new1 = np.zeros(img.shape)
    #在原始图像周围填充像素(要填充的对象，((a上,a下),(a左,a右)),填充方式)
    #a上下左右，分别为，在图像上下左右填充的数目
    #返回值为填充后的图像
    img_pad = np.pad(img,((dim//2,dim//2),(dim//2,dim//2)),'constant')
    #print(img_pad)
    for i in range(dst_x):
        for j in range(dst_y):
            img_new1[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
    plt.figure(1)#创建一个绘图窗口
    plt.imshow(img_new1.astype(np.uint8),cmap = 'gray')
    #plt.show()

    #2求梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])#横向
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])#纵向

    img_new_tidu_x = np.zeros(img_new1.shape)#横向
    img_new_tidu_y = np.zeros(img_new1.shape)#纵向
    img_new_tidu = np.zeros(img_new1.shape)#横向纵向结合
    #填充边缘,因为sobel算子是三行三列，当中心在图像左上角是只有一行，或一列没有可用来做卷积的数据，所以填充1个
    img_pad = np.pad(img_new1,((1,1),(1,1)),'constant')
    #做卷积运算
    for i in range(dst_x):
        for j in range(dst_y):
            img_new_tidu_x[i,j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_x)#横向
            img_new_tidu_y[i,j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_y)#纵向
            img_new_tidu[i,j] = np.sqrt(img_new_tidu_x[i,j]**2 + img_new_tidu_y[i,j]**2)#横纵结合

    plt.figure(2)
    plt.imshow(img_new_tidu.astype(np.uint8),cmap = 'gray')
    #plt.show()

    #3.非极大值抑制：https://www.cnblogs.com/techyan1990/p/7291771.html
    img_new_fei_jidazhi_yizhi = np.zeros(img_new_tidu.shape)#创建一个空白图像
    img_new_tidu_x[img_new_tidu_x==0] = 0.00000001
    angle = img_new_tidu_y / img_new_tidu_x
    for i in range(1,dst_x-1):
        for j in range(1,dst_y-1):
            flag = False
            G = img_new_tidu[i-1:i+2,j-1:j+2]#目标点(Xi,Yj)的八邻域点,区间是左闭右开
            if angle[i,j] > 0:
                Gup = (1-angle[i,j])*G[1,2] + angle[i,j]*G[0,2]
                Gdown = (1-angle[i,j])*G[1,0] + angle[i,j]*G[2,0]
                if (img_new_tidu[i,j] > Gup and img_new_tidu[i,j] > Gdown):
                    flag = True

            elif angle[i,j] < 0:
                Gup = (1-angle[i,j])*G[1,0] + angle[i,j]*G[0,0]
                Gdown = (1-angle[i,j])*G[1,2] + angle[i,j]*G[2,2]
                if  (img_new_tidu[i,j] > Gup and img_new_tidu[i,j] > Gdown):
                    flag = True

            elif angle[i,j] >= 1:
                Gup = (1-1/angle[i,j])*G[0,1] + (1/angle[i,j])*G[0,2]
                Gdown = (1-1/angle[i,j])*G[2,1] + (1/angle[i,j])*G[2,0]
                if (img_new_tidu[i,j] > Gup and img_new_tidu[i,j] > Gdown):
                    flag = True

            elif angle[i,j] <= -1:
                Gup = (1-1/angle[i,j])*G[0,1] + (1/angle[i,j])*G[0,0]
                Gdown = (1-1/angle[i,j])*G[2,1] + (1/angle[i,j])*G[2,2]
                if (img_new_tidu[i,j] > Gup and img_new_tidu[i,j] > Gdown):
                    flag = True

            if flag:
                img_new_fei_jidazhi_yizhi[i,j] = img_new_tidu[i,j]
    plt.figure(3)
    plt.imshow(img_new_fei_jidazhi_yizhi.astype(np.uint8),cmap = 'gray')
    #plt.show()

    #4双阈值检测，连接边缘
    lower_threshold = img_new_tidu.mean() * 0.5
    high_threshold = lower_threshold * 3
    a_x ,a_y = img_new_fei_jidazhi_yizhi.shape
    stack = []
    for i in range(1,img_new_fei_jidazhi_yizhi.shape[0]-1):
        for j in range(1,img_new_fei_jidazhi_yizhi.shape[1]-1):
            if img_new_fei_jidazhi_yizhi[i,j] >= high_threshold:
                img_new_fei_jidazhi_yizhi[i,j] = 255#置白
                stack.append([i,j])
            elif img_new_fei_jidazhi_yizhi[i,j]<=lower_threshold:
                img_new_fei_jidazhi_yizhi[i,j] = 0#置黑

    while not len(stack) == 0:
        temp_1,temp_2 = stack.pop()
        a = img_new_fei_jidazhi_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]#八邻域
        if (a[0,0]<high_threshold) and (a[0,0]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1-1,temp_2-1] = 255
            stack.append([temp_1-1,temp_2-1])
        if (a[0,1]<high_threshold) and (a[0,1]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1-1,temp_2] = 255
            stack.append([temp_1-1,temp_2])
        if (a[0,2]<high_threshold) and (a[0,2]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1-1,temp_2+1] = 255
            stack.append([temp_1-1,temp_2+1])
        if (a[1,0]<high_threshold) and (a[1,0]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1,temp_2-1] = 255
            stack.append([temp_1,temp_2-1])
        if (a[1,2]<high_threshold) and (a[1,2]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1,temp_2+1] = 255
            stack.append([temp_1,temp_2+1])
        if (a[2,0]<high_threshold) and (a[2,0]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1+1,temp_2-1] = 255
            stack.append([temp_1+1,temp_2-1])
        if (a[2,1]<high_threshold) and (a[2,1]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1+1,temp_2] = 255
            stack.append([temp_1+1,temp_2])
        if (a[2,2]<high_threshold) and (a[2,2]>lower_threshold):
            img_new_fei_jidazhi_yizhi[temp_1+1,temp_2+1] = 255
            stack.append([temp_1+1,temp_2+1])

    for i in range(img_new_fei_jidazhi_yizhi.shape[0]):#行
        for j in range(img_new_fei_jidazhi_yizhi.shape[1]):#列
            if img_new_fei_jidazhi_yizhi[i,j] != 0 and img_new_fei_jidazhi_yizhi[i,j] != 255:
                img_new_fei_jidazhi_yizhi[i,j] = 0


    plt.figure(4)#创建第四个窗口
    plt.imshow(img_new_fei_jidazhi_yizhi.astype(np.uint8),cmap = 'gray')
    plt.show()
