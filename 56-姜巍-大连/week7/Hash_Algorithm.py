import numpy as np
import cv2


def hash_algorithm(img, mode='average'):
    """
    我们可以使用均值哈希算法(Average Hash)或者差值哈希算法(Difference Hash)
    :param img: 传入待处理图像
    :param mode: 选择使用何种哈希算法:'average'为均值哈希算法，'difference'为差值哈希算法
    :return: 哈希值(64位)
    """

    # 1. 对图像进行缩放至预定大小
    if mode == 'average':
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    elif mode == 'difference':
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    else:
        print(f"mode '{mode}' is not included.")

    # 2. 灰度化(如果想分开处理RGB，则接下来的代码需要另外编写)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 计算哈希值
    if mode == 'average':
        average_hash = np.where(img >= np.mean(img), 1, 0).flatten()
        return average_hash
    elif mode == 'difference':
        difference_hash = np.where(img[:, :8] > img[:, 1:], 1, 0).flatten()
        return difference_hash


# 传入图像
image_01 = cv2.imread('lenna.png')
image_02 = cv2.imread('lenna_noise.png')

# 计算均值哈希
aver_hash_img_01 = hash_algorithm(image_01, mode='average')
print(f"'lenna'的均值哈希值为：\n {aver_hash_img_01}")
aver_hash_img_02 = hash_algorithm(image_02, mode='average')
print(f"'lenna_noise'的均值哈希值为：\n {aver_hash_img_02}")

# 判断均值哈希下两幅图的汉明距离(相似度)
aver_hash_hamming_distance = np.sum(aver_hash_img_01 ^ aver_hash_img_02)
print(f"均值哈希计算两幅图汉明距离为{aver_hash_hamming_distance}.")

# 计算插值哈希
diff_hash_img_01 = hash_algorithm(image_01, mode='difference')
print(f"'lenna'的差值哈希值为：\n {diff_hash_img_01}")
diff_hash_img_02 = hash_algorithm(image_02, mode='difference')
print(f"'lenna_noise'的差值哈希值为：\n {diff_hash_img_02}")

# 判断差值哈希下两幅图的汉明距离(相似度)
diff_hash_hamming_distance = np.sum(diff_hash_img_01 ^ diff_hash_img_02)
print(f"差值哈希计算两幅图汉明距离为{diff_hash_hamming_distance}.")
