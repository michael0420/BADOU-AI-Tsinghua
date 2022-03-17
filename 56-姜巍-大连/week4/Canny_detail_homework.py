import numpy as np
import matplotlib.pyplot as plt


class CannyDetail:
    """Canny Detail实现，包含：灰度化、高斯滤波、sobel边缘检测、对梯度幅值进行NMS和双边缘检测"""

    def __init__(self, pic_name: str):
        """属性初始化"""

        self.pic_path = pic_name  # 存储完整文件名
        self.image_src = plt.imread(self.pic_path).astype(np.float64)
        self.image_gray = np.zeros((self.image_src.shape[0], self.image_src.shape[1]))
        self.padding_width = 0
        self.kernel_size = 0
        self.higher_threshold = 0
        self.lower_threshold = 0

    def gray(self):
        """数字图像灰度化"""

        if self.pic_path[-4:] == '.png':  # 判断文件扩展名，如果是.png，则图片在这里的存储格式是0到1的浮点数
            self.image_src = self.image_src * 255  # 注意！这一步完成后img仍然是浮点数类型
        # 浮点法灰度化，运用numpy方法合并RGB通道
        self.image_gray = 0.11 * self.image_src[:, :, 0] + 0.59 * self.image_src[:, :, 1] + 0.3 * self.image_src[:, :, 2]
        # 下面这一条是老师用的均值灰度化代码，用以复现展示效果
        # self.image_gray = self.image_src.mean(axis=-1)

    def gauss_filter_producer(self, sigma=1.0):
        """生成高斯滤波器"""

        radius = (lambda sig: 1 if int(np.round(3 * sig)) < 1 else int(np.round(3 * sig)))(sigma)  # 计算滤波器半径
        print(radius)
        # 上式之所以令滤波器半径接近3倍sigma，是根据统计学原理，使滤波器尺寸覆盖约99.73%的分布范围。
        self.padding_width = radius  # 对可能的padding工作赋予初值
        array_x = np.linspace(-radius, radius, 2 * radius + 1)
        x, y = np.meshgrid(array_x, array_x)  # 用于生成2组数组,x和y对应滤波器格子坐标(滤波器中心坐标(0,0))
        build_filter = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # 根据高斯公式求得滤波器中各个坐标的值
        build_filter /= np.sum(np.sum(build_filter))  # 别忘了前置系数
        print(f"指定滤波器大小为:{2 * radius + 1}×{2 * radius + 1}\n标准差设定为:{sigma}")
        return build_filter

    def zero_padding_operation(self):
        """高斯滤波前也许需要zero padding工作"""

        img_copy = np.zeros(
            (2 * self.padding_width + self.image_gray.shape[0], 2 * self.padding_width + self.image_gray.shape[1]))
        img_copy[self.padding_width:self.padding_width + self.image_gray.shape[0],
        self.padding_width:self.padding_width + self.image_gray.shape[1]] = self.image_gray.copy()
        return img_copy

    def filtrating_operation(self, filt, pad_img):
        """进行高斯滤波"""

        self.kernel_size = 2 * self.padding_width + 1
        img_filtrated = np.zeros(self.image_gray.shape)
        for h in range(img_filtrated.shape[0]):
            for w in range(img_filtrated.shape[1]):
                img_filtrated[h, w] = np.sum(pad_img[h:h + self.kernel_size, w:w + self.kernel_size] * filt[:, :])
        return img_filtrated

    def sobel_difference(self, input_img):
        """使用sobel算子估算xy方向平均梯度，计算梯度大小和方向"""

        # 指定sobel算子
        sobel_w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # 声明存储梯度信息的3通道数组，我们约定通道0 1 2 3依次存储h方向梯度的模、w方向梯度的模、total梯度角度tan值和total梯度的模
        first_difference = np.zeros((input_img.shape[0], input_img.shape[1], 4))
        # 对传入的矩阵进行1×1 padding
        original_padding = np.pad(input_img, ((1, 1), (1, 1)), 'constant')
        # 进行sobel滤波
        for h in range(input_img.shape[0]):
            for w in range(input_img.shape[1]):
                first_difference[h, w, 0] = np.sum(original_padding[h:h + 3, w:w + 3] * sobel_h)
                # 由于需要计算合梯度(total)角度的tan值，因此对x方向为0的梯度，为了程序运行不出错，将其赋值为0.00000001
                first_difference[h, w, 1] = (lambda hh, ww: 0.00000001 if np.sum(
                    original_padding[hh:hh + 3, ww:ww + 3] * sobel_w) == 0 else np.sum(
                    original_padding[hh:hh + 3, ww:ww + 3] * sobel_w))(h, w)
        # 计算合梯度(total)角度的tan值
        first_difference[:, :, 2] = first_difference[:, :, 0] / first_difference[:, :, 1]
        # 计算合梯度(total)的模
        first_difference[:, :, 3] = np.sqrt(first_difference[:, :, 0] ** 2 + first_difference[:, :, 1] ** 2)
        return first_difference

    def non_max_suppressed(self, img_sobel):
        """对sobel滤波后存储的梯度矩阵进行非极大值抑制，存储更加精确的边界"""

        img_nm = img_sobel[:, :, -1].copy()
        # 对于任意点A。以A点为中心的3×3邻域中，在A点存储的梯度方向上，一前一后两个相邻点与A点值比较。梯度方向上没有点，则需要进行插值求虚拟点。
        # 因此，A点去心邻域的8个点与A点连线分割成的8块角度区域，中心对称的区域为同一个梯度方向的区域，所以合并为4个区域讨论(w正方向为angel = 0)
        for h in range(1, img_nm.shape[0] - 1):
            for w in range(1, img_nm.shape[1] - 1):
                # 3 × 3 点阵坐标为：
                #                       w-1  w  w+1
                #                   h-1
                #                   h
                #                   h+1
                # 插值时注意tanθ的符号
                # 首先讨论虚拟点落在h-1和h+1行的情况
                # (situation 1) angel在区间[-π/2,-π/4] ∪ [π/2,π3/4]
                if img_sobel[h, w, 2] <= -1:
                    behind = img_sobel[h - 1, w, -1] + (img_sobel[h - 1, w, -1] - img_sobel[h - 1, w - 1, -1]) / \
                             img_sobel[h, w, 2]
                    front = img_sobel[h + 1, w, -1] + (img_sobel[h + 1, w, -1] - img_sobel[h + 1, w + 1, -1]) / \
                            img_sobel[h, w, 2]
                    delta = img_sobel[h, w, -1] > max(behind, front)  # 防止代码过长，将判别式单独存在一个变量中
                    # 极大值点则赋值自身，非极大值点赋值0
                    img_nm[h, w] = (lambda d, hh, ww: img_sobel[hh, ww, -1] if d else 0)(delta, h, w)
                # # (situation 2) angel在区间[-π3/4,-π/2] ∪ [π/4,π/2]
                elif img_sobel[h, w, 2] >= 1:
                    behind = img_sobel[h - 1, w, -1] + (img_sobel[h - 1, w + 1, -1] - img_sobel[h - 1, w, -1]) / \
                             img_sobel[h, w, 2]
                    front = img_sobel[h + 1, w, -1] + (img_sobel[h + 1, w - 1, -1] - img_sobel[h + 1, w, -1]) / \
                            img_sobel[h, w, 2]
                    delta = img_sobel[h, w, -1] > max(front, behind)
                    img_nm[h, w] = (lambda d, hh, ww: img_sobel[hh, ww, -1] if d else 0)(delta, h, w)
                # # 然后讨论虚拟点落在i-1和i+1列的情况
                # # (situation 3) angel在区间[-π,-π3/4] ∪ [0,π/2]
                elif img_sobel[h, w, 2] > 0:
                    behind = img_sobel[h, w - 1, -1] + (img_sobel[h + 1, w - 1, -1] - img_sobel[h, w - 1, -1]) * \
                             img_sobel[h, w, 2]
                    front = img_sobel[h, w + 1, -1] + (img_sobel[h + 1, w + 1, -1] - img_sobel[h, w + 1, -1]) * \
                            img_sobel[h, w, 2]
                    delta = img_sobel[h, w, -1] > max(front, behind)
                    img_nm[h, w] = (lambda d, hh, ww: img_sobel[hh, ww, -1] if d else 0)(delta, h, w)
                # # (situation 4) angel在区间[-π/2,0] ∪ [π3、4,π]
                elif img_sobel[h, w, 2] < 0:
                    behind = img_sobel[h, w - 1, -1] + (img_sobel[h, w - 1, -1] - img_sobel[h - 1, w - 1, -1]) * \
                             img_sobel[h, w, 2]
                    front = img_sobel[h, w + 1, -1] + (img_sobel[h, w + 1, -1] - img_sobel[h + 1, w + 1, -1]) * \
                            img_sobel[h, w, 2]
                    delta = img_sobel[h, w, -1] > max(front, behind)
                    img_nm[h, w] = (lambda d, hh, ww: img_sobel[hh, ww, -1] if d else 0)(delta, h, w)
        return img_nm

    def double_threshold_check(self, img_nms, img_sobel, low_threshold=0, high_threshold=255):
        """双阈值边缘检测"""

        # 首先设置阈值
        self.lower_threshold = low_threshold
        self.higher_threshold = high_threshold
        # 以下是为了复现老师的代码写的，验证后可以删除
        # self.lower_threshold = img_sobel[:, :, -1].mean() * 0.5
        # self.higher_threshold = self.lower_threshold * 3
        # 根据高低阈值将NMS矩阵中的点进行分类：噪声、弱边缘和强边缘，这里用到堆栈(stack)存储弱边缘点坐标
        img_double_th = img_nms.copy()
        img_double_th = np.where(img_double_th >= self.higher_threshold, 255, img_double_th)  # 强边缘赋值255
        img_double_th = np.where(img_double_th <= self.lower_threshold, 0, img_double_th)  # 噪声点赋值0，剩下的就是弱边缘
        # 声明一个空栈，进行坐标存储
        stack = [[h, w] for h in range(1, img_double_th.shape[0] - 1) for w in range(1, img_double_th.shape[1] - 1) if img_double_th[h, w] == 255]
        # 进行强边缘周围是否存在边缘的检测，如果N8(p)邻域存在弱边缘，则该弱边缘荣升为强边缘
        while len(stack) != 0:
            temp_h, temp_w = stack.pop()  # 出栈
            # 选取以[temp_h, temp_w]为中心的3×3邻域
            bridge_array = img_double_th[temp_h - 1:temp_h + 2, temp_w - 1:temp_w + 2]
            # 将邻域中的弱边缘点赋值255，变成强边缘点
            bridge_array_01 = np.where(bridge_array >= self.lower_threshold, 255, bridge_array)
            bridge_array_02 = np.where(bridge_array <= self.higher_threshold, 255, bridge_array)
            bridge_array = np.where(bridge_array_01 == bridge_array_02, 255, bridge_array)
            # 获取邻域中强边缘点坐标，存储在列表中
            x, y = np.where(bridge_array == 255)
            # 将可能入栈的图像边界上的坐标点去除(避免下次循环时出现错误导致程序无法进行下去)
            a = [0, img_double_th.shape[0]]  # h方向边图像界点坐标值
            b = [0, img_double_th.shape[1]]  # w方向边图像界点坐标值
            new_temp_hw = [[i, j] for i, j in zip(x, y) if (i not in a and j not in b)]
            # 列表入栈
            stack += new_temp_hw
            # 将重复入栈的[temp_h, temp_w]剔除
            if [temp_h, temp_w] in stack:
                stack.remove([temp_h, temp_w])

        # 将NMS图像中剩余的点(不与强边缘连接的游离点或者大片弱边缘)赋值0
        img_bridge_01 = np.where(img_double_th > 0, 0, img_double_th)
        img_bridge_02 = np.where(img_double_th < 255, 0, img_double_th)
        img_double_th = np.where(img_bridge_01 == img_bridge_02, 0, img_double_th)
        return img_double_th


project01 = CannyDetail('lenna.png')
project01.gray()  # 灰度化
filter_receiver = project01.gauss_filter_producer(0.5)  # 生成高斯滤波器
zero_padding_copy = project01.zero_padding_operation()  # padding扩展图像尺寸
filtrated_img = project01.filtrating_operation(filter_receiver, zero_padding_copy)  # 对图片进行高斯滤波平滑
sobeled_img = project01.sobel_difference(filtrated_img)  # 使用sobel算子处理图像，依次返回hw方向的梯度模、角度和合梯度的模
non_max_img = project01.non_max_suppressed(sobeled_img)  # 对sobel滤波后存储的梯度矩阵进行非极大值抑制，存储更加精确的边界
double_threshold_img = project01.double_threshold_check(non_max_img, sobeled_img, 50, 100)  # 双阈值检测优化边界

# 画图部分
plt.figure(1)
plt.imshow(project01.image_src.astype(np.uint8))
plt.axis('off')
plt.title("lenna_src", fontsize='small')

plt.figure(2)
plt.imshow(project01.image_gray, cmap='gray')
plt.axis('off')
plt.title("lenna_gray", fontsize='small')

plt.figure(3)
plt.imshow(filtrated_img, cmap='gray')
plt.axis('off')
plt.title("lenna_filtrated", fontsize='small')

plt.figure(4)
plt.imshow(sobeled_img[:, :, -1].astype(np.uint8), cmap='gray')  # 需要对合向量的模做数据类型转换才能变成[0,255]的整型数据
plt.axis('off')
plt.title("lenna_sobeled", fontsize='small')

plt.figure(5)
plt.imshow(non_max_img.astype(np.uint8), cmap='gray')  # 需要对合向量的模做数据类型转换才能变成[0,255]的整型数据
plt.axis('off')
plt.title("lenna_NMS", fontsize='small')

plt.figure(6)
plt.imshow(double_threshold_img.astype(np.uint8), cmap='gray')  # 需要对合向量的模做数据类型转换才能变成[0,255]的整型数据
plt.axis('off')
plt.title("lenna_DT", fontsize='small')

plt.show()
