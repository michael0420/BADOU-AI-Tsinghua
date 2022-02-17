import numpy as np
import cv2

class Canny():
    def __init__(self, sigma, thred1=200, thred2=300):
        self.img = cv2.imread('/home/uers/desk_B/八斗/week2/lenna.png')
        self.shape = self.img.shape[:2]
        self.gray = []
        self.gray_gausian = []
        self.img_grad = []
        self.angle = []
        self.img_nms = []

        self.gray = self.img2gray()
        self.gray_gausian = self.gaussian_filter(sigma)
        self.img_grad, self.angle = self.img2grad()
        self.img_nms = self._nms()
        self.img_nms = self.edge_connect(thred1,thred2)

    def img2gray(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        print("finish bgr to gray!")
        return gray

    def gaussian_filter(self,sigma):
        import math
        dim = int(np.round(8*sigma+1))
        print("gaussian filter dim is: ", dim)
        filter = np.zeros([dim, dim])
        tmp = [i-dim//2 for i in range(dim)]
        for i in range(dim):
            for j in range(dim):
                filter[i,j] = (1/2*math.pi*sigma**2)*math.exp(-(tmp[i]**2+tmp[j]**2)/(2*sigma**2))
        filter = filter/filter.sum()
        img_new = np.zeros(self.shape)
        tmp = dim//2
        img_pad = np.pad(self.gray, ((dim,dim),(dim,dim)), 'constant')
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*filter)
        print("finish gaussian filter!")
        return img_new

    def img2grad(self):
        sobel_kernal_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_kernal_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        img_grad_x = np.zeros(self.gray.shape)
        img_grad_y = np.zeros(self.gray.shape)
        img_grad = np.zeros(self.gray.shape)
        img_pad = np.pad(self.gray_gausian,((1, 1), (1, 1)), 'constant')
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                img_grad_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernal_x)
                img_grad_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernal_y)
                img_grad[i,j] = np.sqrt(img_grad_x[i,j]**2 + img_grad_y[i,j]**2)
        img_grad_x[img_grad_x == 0] = 0.00000001
        angle = img_grad_y / img_grad_x
        print("finish img to grad!")
        plt.figure(2)
        plt.imshow(img_grad.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return img_grad, angle

    def _nms(self):
        img_nms = np.zeros(self.img_grad.shape)
        for i in range(1,self.img_grad.shape[0]-1):
            for j in range(1,self.img_grad.shape[1]-1):
                tmp = self.img_grad[i-1:i+2, j-1:j+2]
                if self.angle[i,j] <= -1:
                    ang1 = tmp[0,0] + (tmp[0,1] - tmp[0,0])*self.angle[i,j]
                    ang2 = tmp[2,1] + (tmp[2,2] - tmp[2,1])*self.angle[i,j]
                    # ang1 = (tmp[0, 1] - tmp[0, 0]) / angle[i, j] + tmp[0, 1]
                    # ang2 = (tmp[2, 1] - tmp[2, 2]) / angle[i, j] + tmp[2, 1]
                    if self.img_grad[i,j] > ang1 and self.img_grad[i,j] > ang2:
                        img_nms[i,j] = self.img_grad[i,j]
                elif self.angle[i,j] >=1:
                    ang1 = tmp[0,1] + (tmp[0,2]-tmp[0,1])*self.angle[i,j]
                    ang2 = tmp[2,1] + (tmp[2,0]-tmp[2,1])*self.angle[i,j]
                    if self.img_grad[i,j] > ang1 and self.img_grad[i,j] > ang2:
                        img_nms[i,j] = self.img_grad[i,j]
                elif self.angle[i,j] >0:
                    ang1 = tmp[1,2] + (tmp[0,2]-tmp[1,2])*self.angle[i,j]
                    ang2 = tmp[1,0] + (tmp[2,0]-tmp[1,0])*self.angle[i,j]
                    if self.img_grad[i,j] > ang1 and self.img_grad[i,j] > ang2:
                        img_nms[i,j] = self.img_grad[i,j]
                elif self.angle[i,j] <0:
                    ang1 = tmp[1,2] + (tmp[2,2]-tmp[1,2])*self.angle[i,j]
                    ang2 = tmp[1,0] + (tmp[0,0]-tmp[1,0])*self.angle[i,j]
                    if self.img_grad[i,j] > ang1 and self.img_grad[i,j] > ang2:
                        img_nms[i,j] = self.img_grad[i,j]
        print("finish nms!")
        plt.figure(3)
        plt.imshow(img_nms.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return img_nms

    def edge_connect(self,thred1,thred2):
        low_thred = self.img_grad.mean()
        hign_thred = low_thred*3
        print("low_thred is %d, hign_thred is %d !"%(low_thred, hign_thred))
        edge_zhan = []
        for i in range(1, self.img_grad.shape[0]-1):
            for j in range(1, self.img_grad.shape[1]-1):
                if self.img_nms[i,j] > hign_thred:
                    self.img_nms[i,j] = 255
                    edge_zhan.append([i,j])
                elif self.img_nms[i,j] <= low_thred:
                    self.img_nms[i,j] = 0
        while len(edge_zhan):
            tx, ty = edge_zhan.pop()
            tmp = self.img_nms[tx-1:tx+2, ty-1:ty+2]
            for i in range(3):
                for j in range(3):
                    if tmp[i,j] < hign_thred and tmp[i,j] > low_thred:
                        self.img_nms[tx+i-1, ty+j-1] = 255
                        edge_zhan.append([tx+i-1,ty+j-1])
        for i in range(self.img_nms.shape[0]):
            for j in range(self.img_nms.shape[1]):
                if self.img_nms[i,j] !=0 and self.img_nms[i,j] != 255:
                    self.img_nms[i,j] = 0
        print("finish edge_connect!")
        plt.figure(4)
        plt.imshow(self.img_nms.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return self.img_nms

    def plot_figure(self):
        import matplotlib.pyplot as plt
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure(1)
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title("---image lenna----")
        plt.subplot(2, 3, 2)
        plt.imshow(self.gray, cmap='gray')
        plt.title("---image gray----")
        plt.subplot(2, 3, 3)
        plt.imshow(self.gray_gausian, cmap='gray')
        plt.title("-----imge_gray_gausian------")
        plt.subplot(2, 3, 4)
        plt.imshow(self.img_grad, cmap='gray')
        plt.title("-----imge_grad------")
        plt.subplot(2, 3, 5)
        plt.imshow(self.img_nms)
        plt.title("-----imge_nms------")
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.Canny(self.gray, 200, 300))
        plt.title("-----cv2.Canny------")
        plt.suptitle("Result of Canny")
        plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    canny = Canny(0.5,200,300)
    canny.plot_figure()
