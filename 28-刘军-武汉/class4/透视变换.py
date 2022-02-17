import numpy as np
import cv2
def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0]==dst.shape[0] and src.shape[0]>=4
    nums = src.shape[0]
    A = np.zeros((nums*2,8))
    B = np.zeros((nums*2,1))
    for i in range(nums):
        A[i*2,:] = [src[i,0],src[i,1],1,0,0,0,-src[i,0]*dst[i,0],-src[i,1]*dst[i,0]]
        B[i*2,0] = dst[i,0]
        A[i * 2+1, :] = [0,0,0,src[i,0],src[i,1],1,-src[i,0]*dst[i,1],-src[i,1]*dst[i,1]]
        B[i * 2+1, 0] = dst[i, 1]
    A = np.mat(A)
    warpMatrix = np.dot(A.I,B)
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],1.0,axis=0)
    warpMatrix = warpMatrix.reshape(3,3)
    warpMatrix = np.array(warpMatrix)
    return warpMatrix
def full(new):
    h, w, channels = new.shape
    for i in range(h-1):
        for j in range(w-1):
            if new[i,j,0]==300:
                if i==0 or j==0:
                    new[i,j,:] = [255,255,255]
                new[i,j,:] = new[i-1,j,:]
    return new

def My_warpPerspective(img,warpMatrix,boundary):
    h,w,channels = img.shape
    new = np.zeros((boundary[0]+1,boundary[1]+1,channels),np.uint8)+300
    for i in range(h-1):
        for j in range(w-1):
            org_array = np.array([j,i,1]).reshape(3,1)
            new_array = np.dot(warpMatrix,org_array)
            Y = int(new_array[0,0]/new_array[2,0]+0.5)
            X = int(new_array[1,0]/new_array[2,0] + 0.5)
            if ((X<=boundary[0]) and (X>=0))and((Y<=boundary[1])and (Y>=0)):
                new[X,Y,:] = img[i,j,:]
    new = full(new).astype(np.uint8)
    return new



src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = WarpPerspectiveMatrix(src, dst)

img = cv2.imread('photo1.jpg')
result3 = img.copy()
new1 = cv2.warpPerspective(result3, m, (337, 488))
new2 = My_warpPerspective(result3,m,(488,337))
new3 = cv2.GaussianBlur(new2,(3,3),0)
cv2.imshow('org',img)
cv2.imshow('new1',new1)
cv2.imshow('new2',new2)
cv2.imshow('new3',new3)
cv2.waitKey(0)
