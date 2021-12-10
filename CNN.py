import numpy as np
import cv2
import matplotlib.pyplot as plt


def Kern(Target, Kernel):
    result = 0
    for i in range(Kernel.shape[0]):
        for j in range(Kernel.shape[1]):
            result += Target[i, j] * Kernel[i, j]
    return int(result)


def conv_1(Input, Kernel, Padding, Stride):
    # Given input, kernel matrix and padding, stride figure, work out the target matrix
    # conv_1(array, array, int, int) --> array

    # Starting kernel from input matrix
    k1, k2 = Kernel.shape

    # Target matrix
    rows = int(((Input.shape[0] - Kernel.shape[0] + 2 * Padding) / Stride) + 1)
    cols = int(((Input.shape[1] - Kernel.shape[1] + 2 * Padding) / Stride) + 1)

    # Padding
    Input = np.vstack((np.zeros((Padding, Input.shape[1])), Input))
    Input = np.hstack((np.zeros((Input.shape[0], Padding)), Input))
    Input = np.vstack((Input, np.zeros((Padding, Input.shape[1]))))
    Input = np.hstack((Input, np.zeros((Input.shape[0], Padding))))

    T_mat = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            T_mat[r, c] = Kern(Input[(Stride * r):(Stride * r + k1), (Stride * c):(Stride * c + k2)], Kernel)
    return T_mat


img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
# kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# kernel = np.array([[1/16, 2/16, 1/16], [2/16, 2/16, 2/16], [1/16, 2/16, 1/16]])
# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
print(kernel)

# convolution pic for gray img
cov_img = conv_1(gray, kernel, 0, 1)
print(cov_img)
plt.imshow(cov_img, cmap='gray')
plt.show()

# cv2.imshow("convolution", cov_img)
# cv2.waitKey(0)

# convolution pic for colorful img
# (b, g, r) = cv2.split(img)
# bC = conv_1(b, kernel, 0, 1)
# gC = conv_1(g, kernel, 0, 1)
# rC = conv_1(r, kernel, 0, 1)
# cov_img = cv2.merge((bC, gC, rC))
# cv2.imshow("convolution", cov_img)
# cv2.waitKey(0)