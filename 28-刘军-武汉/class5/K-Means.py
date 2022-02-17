import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png')
h,w,channels = img.shape
img_data = img.reshape(h*w,3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER,20,0.5)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness_dict = {}
labels_dict = {}
centers_dict = {}
img_result = {}
for k in range(1,7):
    compactness, labels, centers = cv2.kmeans(img_data,2**k,None,criteria,10,flags)
    compactness_dict['k值为%d'%(2**k)] = compactness
    labels_dict['k值为%d'%(2**k)] = labels
    centers_dict['k值为%d'%(2**k)] = centers
    img_result['k值为%d'%(2**k)] = cv2.cvtColor(np.uint8(centers)[labels.flatten()].reshape(h,w,channels)
                                              ,cv2.COLOR_BGR2RGB)
plt.rcParams['font.sans-serif'] = 'SimHei'
for i in range(1,7):
    plt.subplot(2,3,i)
    plt.imshow(img_result['k值为%d'%(2**i)])
    plt.title('k值为%d'%(2**i))
plt.show()