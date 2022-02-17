import cv2
import numpy as np

class WarpPerspective():
    def __init__(self,src,dst):
        self.img = cv2.imread('/home/uers/desk_B/八斗/week3/photo1.jpg')
        self.src = src
        self.dst = dst

    def getPerspectiveMatrix(self):
        assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
        num = src.shape[0]
        A = np.zeros((2*num,8))
        B = np.zeros((2*num,1))
        for i in range(num):
            a = src[i,:]
            b = dst[i,:]
            A[2*i,:] = [a[0], a[1], 1, 0, 0, 0, -a[0]*b[0], -a[1]*b[0]]
            A[2*i+1,:] = [0, 0, 0, a[0], a[1], 1, -a[0]*b[1], -a[1]*b[1]]
            B[2*i] = b[0]
            B[2*i+1] = b[1]
        A = np.mat(A)
        warpMatrix = A.I * B
        warpMatrix = warpMatrix.T[0]
        warpMatrix = np.insert(warpMatrix, warpMatrix.shape[1], values=1.0, axis=1)
        warpMatrix = warpMatrix.reshape((3,3))
        return warpMatrix

    def warpPerspective(self,m):
        return cv2.warpPerspective(self.img.copy(), m, (337, 488))


if __name__ == "__main__":
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    WP = WarpPerspective(src,dst)
    m = WP.getPerspectiveMatrix()
    print(m)
    result = WP.warpPerspective(m)
    import matplotlib.pyplot as plt
    img = cv2.cvtColor(WP.img, cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("---oral image----")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("---warp image----")
    plt.show()