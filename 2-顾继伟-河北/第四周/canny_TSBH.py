import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class canny_TSBH():
    def canny_Interface(self):
        img = cv.imread("lenna.png", 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("canny_Interface", cv.Canny(gray, 200, 300))
        cv.waitKey(0)
        cv.destroyAllWindows()
    def canny_Detail(self):
        pic_Path = "lenna.png"
        img = plt.imread(pic_Path)
        if pic_Path[-4:] == '.png':
            img = img*255
        img = img.mean(axis=-1)

        # No.1 Gaussian smoothing
        sigma = 0.5
        dim = int(np.round(6*sigma + 1))
        if dim%2 == 0:
            dim+=1
        gs_Filter = np.zeros([dim, dim])
        tmp = [i-dim//2 for i in range(dim)]
        n1 = 1/(2 * np.math.pi * sigma ** 2)
        n2 = -1/(2*sigma**2)
        for i in range(dim):
            for j in range(dim):
                gs_Filter[i,j] = n1*np.math.exp(n2*(tmp[i]**2+tmp[j]**2))
        gs_Filter = gs_Filter/gs_Filter.sum()
        dx, dy = img.shape
        img_New = np.zeros(img.shape)
        tmp = dim//2
        img_Pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
        for i in range(dx):
            for j in range(dy):
                img_New[i, j] = np.sum(img_Pad[i:i+dim, j:j+dim]*gs_Filter)
        plt.figure(1)
        plt.imshow(img_New.astype(np.uint8), cmap = 'gray')
        plt.axis('off')

        # No.2 Compute gradient
        sobel_Kernel_X = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
        sobel_Kernel_Y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
        img_Tidu_X = np.zeros(img_New.shape)
        img_Tidu_Y = np.zeros([dx, dy])
        img_Tidu = np.zeros(img_New.shape)
        img_Pad = np.pad(img_New, ((1,1), (1,1)), 'constant')
        for i in range(dx):
            for j in range(dy):
                img_Tidu_X[i, j] = np.sum(img_Pad[i:i+3, j:j+3]*sobel_Kernel_X)
                img_Tidu_Y[i, j] = np.sum(img_Pad[i:i+3, j:j+3]*sobel_Kernel_Y)
                img_Tidu[i, j] = np.sqrt(img_Tidu_X[i, j]**2 + img_Tidu_Y[i, j]**2)
        img_Tidu_X[img_Tidu_X == 0] = 0.00000001
        angle = img_Tidu_Y/img_Tidu_X
        plt.figure(2)
        plt.imshow(img_Tidu.astype(np.uint8), cmap = 'gray')
        plt.axis('off')

        # No.3 Non-maximum suppression
        img_Max_limit = np.zeros(img_Tidu.shape)
        for i in range(1, dx-1):
            for j in range(1, dy-1):
                flag = True
                temp = img_Tidu[i-1:i+2, j-1:j+2]
                if angle[i,j] <= -1:
                    num_1 = (temp[0,1]-temp[0,0])/angle[i,j] + temp[0,1]
                    num_2 = (temp[2,1]-temp[2,2])/angle[i,j] + temp[2,1]
                    if not (img_Tidu[i,j]>num_1 and img_Tidu[i,j]>num_2):
                        flag = False
                elif angle[i,j]>=1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                    if not (img_Tidu[i, j] > num_1 and img_Tidu[i, j] > num_2):
                        flag = False
                elif angle[i,j]>=0:
                    num_1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                    if not (img_Tidu[i, j] > num_1 and img_Tidu[i, j] > num_2):
                        flag = False
                elif angle[i,j]<0:
                    num_1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                    if not (img_Tidu[i, j] > num_1 and img_Tidu[i, j] > num_2):
                        flag = False
                if flag:
                    img_Max_limit[i,j] = img_Tidu[i,j]
        plt.figure(3)
        plt.imshow(img_Max_limit.astype(np.uint8), cmap = 'gray')
        plt.axis('off')

        # No.4 Double threshold detection
        low_Boundary = img_Tidu.mean()*0.5
        high_Boundary = low_Boundary*3
        stack = []
        for i in range(1, img_Max_limit.shape[0]-1):
            for j in range(1, img_Max_limit.shape[1]-1):
                if img_Max_limit[i,j] >= high_Boundary:
                    img_Max_limit[i,j] = 255
                    stack.append([i,j])
                elif img_Max_limit[i,j] <= low_Boundary:
                    img_Max_limit[i,j] = 0
        while not len(stack) == 0:
            temp1, temp2 = stack.pop()
            a = img_Max_limit[temp1 - 1:temp1 + 2, temp2 - 1:temp2 + 2]
            if ((a[0, 0] < high_Boundary) and (a[0, 0]) > low_Boundary):
                img_Max_limit[temp1 - 1, temp2 - 1] = 255
                stack.append([temp1 - 1, temp2 - 1])
            if ((a[0, 1] < high_Boundary) and (a[0, 1]) > low_Boundary):
                img_Max_limit[temp1 - 1, temp2] = 255
                stack.append([temp1 - 1, temp2])
            if ((a[0, 2] < high_Boundary) and (a[0, 2]) > low_Boundary):
                img_Max_limit[temp1 - 1, temp2 + 1] = 255
                stack.append([temp1 - 1, temp2 + 1])
            if ((a[1, 0] < high_Boundary) and (a[1, 0]) > low_Boundary):
                img_Max_limit[temp1, temp2 - 1] = 255
                stack.append([temp1, temp2 - 1])
            if ((a[1, 2] < high_Boundary) and (a[1, 2]) > low_Boundary):
                img_Max_limit[temp1, temp2 + 1] = 255
                stack.append([temp1, temp2 + 1])
            if ((a[2, 0] < high_Boundary) and (a[2, 0]) > low_Boundary):
                img_Max_limit[temp1 + 1, temp2 - 1] = 255
                stack.append([temp1 + 1, temp2 - 1])
            if ((a[2, 1] < high_Boundary) and (a[2, 1]) > low_Boundary):
                img_Max_limit[temp1 + 1, temp2] = 255
                stack.append([temp1 + 1, temp2])
            if ((a[2, 2] < high_Boundary) and (a[2, 2]) > low_Boundary):
                img_Max_limit[temp1 + 1, temp2 + 1] = 255
                stack.append([temp1 + 1, temp2 + 1])

        for i in range(img_Max_limit.shape[0]):
            for j in range(img_Max_limit.shape[1]):
                if img_Max_limit[i, j] != 0 and img_Max_limit[i, j] != 255:
                    img_Max_limit[i, j] = 0

        plt.figure(4)
        plt.imshow(img_Max_limit.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()

    def TouShiBianHuan(self):
        img = cv.imread("photo1.jpg")
        result_Img = img.copy()
        src = np.float32([[207,151],[517,285],[17,601],[343,731]])
        dst = np.float32([[0,0],[337,0],[0,488],[337,488]])
        print(img.shape)

        m = cv.getPerspectiveTransform(src,dst)
        print("wrapMatrix:")
        print(m)
        result = cv.warpPerspective(result_Img,m,(337,488))
        cv.imshow("src",img)
        cv.imshow("dst",result)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def warpMatrix(self,src,dst):
        assert src.shape[0] == dst.shape[0] and src.shape[0]>=4
        nums = src.shape[0]
        a = np.zeros((2*nums,8))
        b = np.zeros((2*nums,1))
        for i in range(0,nums):
            ai = src[i,:]
            bi = dst[i,:]
            a[2*i,:] = [ai[0],ai[1],1,0,0,0,-ai[0]*bi[0],-ai[1]*bi[0]]
            b[2*i] = bi[0]
            a[2*i+1,:] = [0,0,0,ai[0],ai[1],1,-ai[0]*bi[1],-ai[1]*bi[1]]
            b[2*i+1] = bi[1]
        a = np.mat(a)
        warpMatrix = a.I *b
        warpMatrix = np.array(warpMatrix).T[0]
        warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0)
        warpMatrix = warpMatrix.reshape((3,3))
        return warpMatrix








if __name__ == '__main__':
    # canny_TSBH().canny_Interface()
    # canny_TSBH().canny_Detail()
    # canny_TSBH().TouShiBianHuan()

    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    dst = [[46.0,920.0],[46.0,100.0],[600.0,100.0],[600.0,920.0]]
    dst = np.array(dst)
    warpMtrix = canny_TSBH().warpMatrix(src, dst)
    print(warpMtrix)
