'''
@auther:Jelly
@用于实现透视变换

程序输出：
    每输出一张图片按任意建继续：

    原始图片
    Canny边缘检测图片
    霍夫变换后直线图片
    顶点标记图片
    透视变换后图片

相关接口：
    def getPerspectiveTransform_self(src,dst)
    函数用途：用于获得透视变换的变换矩阵
    函数参数：
        sec:原始坐标，个数大于等于4
        dst:变换坐标，个数大于等于4

    def warpPerspective_self(img,Transform_matrix,shape)
    函数用途：利用变换矩阵实现透视变换
    函数参数：
        img: 输入原始图片
        Transform_matrix：透视变换输入矩阵
        shape：输入图片的大小

    def get_Img_Point(img)
    函数用途：获得图片透视变换目标的顶点坐标

'''
import numpy as np
import cv2


def getPerspectiveTransform_self(src,dst):
    '''
    函数用途：用于获得透视变换的变换矩阵
    函数参数：
        sec:原始坐标，个数大于等于4
        dst:变换坐标，个数大于等于4
    '''
    #assert 无需等待程序崩溃打印错误信息，保证输出的点数维度相同，且个数大于4
    assert src.shape[0]==dst.shape[0] and src.shape[0]>=4
    nums = src.shape[0]
    A = np.zeros((2*nums,8))
    B = np.zeros((2*nums,1))
    for i in range(0,nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        #获得推导后的矩阵
        A[2*i,:] = [A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2 * i+1, :] = [0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2 * i+1] = B_i[1]
    #将A数组转换为矩阵（若A本身为矩阵则生成一个新的引用）
    A = np.mat(A)
    #A.I求A矩阵的转置 a*w=b -> w=a-1*b
    warpMatrix = np.dot(A.I,B)
    #对warpMatrix作处理
    warpMatrix = np.array(warpMatrix).T[0]
    #将最后一个数据补充为1
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0)
    warpMatrix = warpMatrix.reshape((3,3))
    return warpMatrix

def warpPerspective_self(img,Transform_matrix,shape):
   '''
    函数用途：利用变换矩阵实现透视变换
    函数参数：
        img: 输入原始图片
        Transform_matrix：透视变换输入矩阵
        shape：输入图片的大小
   '''
   h,w,axis = img.shape
   Transform_matrix = np.mat(Transform_matrix).I   # 求逆矩阵
   result = np.zeros((shape[1],shape[0],axis),dtype=np.uint8)     # 转换图片是数据大小为np.uint8
   for i in range(shape[1]):
        for j in range(shape[0]):
            ZB = np.dot(Transform_matrix,np.mat([j,i,1]).T)
            ZB = ZB/ZB[2,0]
            if (ZB[0,0]<=w and ZB[0,0]>=0) and (ZB[1,0]<=h and ZB[1,0]>=0):
                for ax in range(axis):
                    result[i,j,ax] = img[round(ZB[1,0]),round(ZB[0,0]),ax]
   return result



def get_Img_Point(img):

    # 将原图转为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Canny边缘检测
    canny_img = cv2.Canny(gray_img, 100, 120, 3)
    # 显示边缘检测后的图像
    cv2.imshow("canny_img", canny_img)
    cv2.waitKey(0)
    # #Hough直线检测
    lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 70, minLineLength=180, maxLineGap=30)
    # 得到直线坐标 x1,y1,x2,y2

    def draw_line(img, lines):
        # 绘制直线
        for line_points in lines:
            cv2.line(img, (line_points[0][0], line_points[0][1]), (line_points[0][2], line_points[0][3]),
                     (0, 255, 0), 2, 8, 0)
        cv2.imshow("line_img", img)
        cv2.waitKey(0)

    # 基于边缘检测的图像来检测直线
    draw_line(img, lines)

    # 计算四条直线在图像上的交点作为顶点坐标
    def computer_intersect_point(lines,Img_shape):
        def get_line_k_b(line_point):
            """计算直线的斜率和截距
            :param line_point: 直线的坐标点
            :return:
            """
            # 获取直线的两点坐标
            x1, y1, x2, y2 = line_point[0]
            # 计算直线的斜率和截距
            k = (y1 - y2) / (x1 - x2)
            b = y2 - x2 * (y1 - y2) / (x1 - x2)
            return k, b

        # 用来存放直线的交点坐标
        line_intersect = []
        # 计算所有直线的交点
        for i in range(len(lines)):
            k1, b1 = get_line_k_b(lines[i])
            for j in range(i + 1, len(lines)):
                k2, b2 = get_line_k_b(lines[j])
                # 计算交点坐标
                x = (b2 - b1) / (k1 - k2)
                y = k1 * (b2 - b1) / (k1 - k2) + b1
                # 在图像内的坐标保留
                if (x > 0 and x <= Img_shape[1]) and (y > 0 and y <= Img_shape[0]):
                    line_intersect.append((int(np.round(x)), int(np.round(y))))
        return line_intersect

    def draw_point(img, points):
        for position in points:
            cv2.circle(img, position, 5, (0, 0, 255), -1)
        cv2.imshow("draw_point", img)
        cv2.waitKey(0)

    # 计算直线的交点坐标
    line_intersect = computer_intersect_point(lines,img.shape)
    # 绘制交点坐标的位置
    draw_point(img, line_intersect)

    def order_point(points):
        """对交点坐标进行排序
        :param points:
        :return:
        """
        points_array = np.array(points)
        # 对x的大小进行排序
        x_sort = np.argsort(points_array[:, 0])
        # 对y的大小进行排序
        y_sort = np.argsort(points_array[:, 1])
        # 获取最左边的顶点坐标
        left_point = points_array[x_sort[0]]
        # 获取最右边的顶点坐标
        right_point = points_array[x_sort[-1]]
        # 获取最上边的顶点坐标
        top_point = points_array[y_sort[0]]
        # 获取最下边的顶点坐标
        bottom_point = points_array[y_sort[-1]]
        return np.array([top_point,right_point,left_point,bottom_point], dtype=np.float32)

    # 对原始图像的交点坐标进行排序
    clockwise_point = order_point(line_intersect)
    return clockwise_point









if __name__ == "__main__":

    img = cv2.imread('photo1.jpg')
    result3 = img.copy()

#    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    src = get_Img_Point(img)   #计算出的点为从顶部、右侧、左侧、底部

    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    warpMatrix = getPerspectiveTransform_self(src,dst)

    m = cv2.getPerspectiveTransform(src, dst)
    result = warpPerspective_self(result3,warpMatrix,(337,488))   #w,h
    cv2.imshow("src", img)
    cv2.imshow("result", result)

    cv2.waitKey(0)