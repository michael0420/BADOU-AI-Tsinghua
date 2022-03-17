import numpy as np
import cv2


class Pretreatment:
    """将透视变换传入的8个坐标转换成系数矩阵"""

    def __init__(self, coordinate_src, coordinate_dst):
        """初始化属性"""

        self.src = coordinate_src
        self.dst = coordinate_dst
        # 根据下列方程组构建系数矩阵，(h,w)来自src，透视变换后的(H,W)指向dst
        # a11 h + a12 w + a13                       - a31 h H - a32 w H = H
        #                       a21 h + a22 w + a23 - a31 h W - a32 w W = W
        self.coefficient = np.zeros((self.src.shape[0] * 2, 8))
        self.b = np.zeros(self.src.shape[0] * 2)

    def build_array(self):
        """构造系数矩阵"""

        for i in range(self.src.shape[0]):
            self.coefficient[i, :] = np.array([self.src[i, 0], self.src[i, 1], 1, 0, 0, 0, -self.src[i, 0] * self.dst[i, 0], -self.src[i, 1] * self.dst[i, 0]])
            self.coefficient[i + 4, :] = np.array([0, 0, 0, self.src[i, 0], self.src[i, 1], 1, -self.src[i, 0] * self.dst[i, 1], -self.src[i, 1] * self.dst[i, 1]])
            self.b[i] = self.dst[i, 0]
            self.b[i + 4] = self.dst[i, 1]
        print(self.coefficient)
        print(self.b)


class CramerLaw:
    """Cramer法则求方程组的解"""

    def __init__(self, coefficient_array, b):
        """初始化属性"""

        self.coeff = coefficient_array  # 方程组系数数组
        self.x = np.zeros((self.coeff.shape[0], 1))  # 存储方程组的解
        self.b = b
        self.det = 0  # 存储系数矩阵行列式值

    def determinant_computation(self):
        """计算系数矩阵行列式的值，判断是否可以使用Cramer法则"""

        self.det = np.linalg.det(self.coeff)  # 求系数矩阵行列式
        if self.det == 0:
            print("系数矩阵行列式等于0，不能使用Cramer法则求解。")
        else:
            print(f"系数矩阵行列式值为{self.det}，方程组有唯一解，可以使用Cramer法则求解。")

    def answer(self):
        """根据xi = Di / |A|，Di为用b替代|A|中第i列中元素所得的行列式"""

        d = np.zeros(self.coeff.shape[0])
        for i in range(len(d)):
            temp_arr01 = self.coeff.copy()
            temp_arr01[:, i] = self.b.T
            d[i] = np.linalg.det(temp_arr01)
        self.x = d / self.det


# 读取需要透视变换图像
img = cv2.imread('photo1.jpg')
image_copy = img.copy()

# 载入原图选取4个参考点坐标，输入目图像4个定位点坐标
src = np.array([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.array([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

# 第一步：使用坐标生成透视变换的系数矩阵等
step01 = Pretreatment(src, dst)
step01.build_array()

# 第二步：使用Cramer法则求得透视变换系数矩阵
step02 = CramerLaw(step01.coefficient, step01.b)
step02.determinant_computation()
step02.answer()

warpMatrix = np.array(list(step02.x) + [1]).reshape(3, 3)
print(warpMatrix)

# 进行透视变换
result = cv2.warpPerspective(image_copy, warpMatrix, (337, 488))  # 这里的输出坐标恰好是目标图像的4个顶点坐标

cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
