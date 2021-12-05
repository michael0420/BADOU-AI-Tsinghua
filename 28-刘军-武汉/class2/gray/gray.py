
# coding: utf-8

# In[1]:


from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import cv2


# In[2]:


img = cv2.imread('lenna.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[3]:


h,w = img_rgb.shape[:2]
img_data = img_rgb.reshape(-1,3)
img_gray=np.array([img_data[:,0]*0.3+img_data[:,1]*0.59+img_data[:,2]*0.11]).reshape(h,w)


# In[4]:


img_binary = np.where((img_gray/255) >= 0.5, 1, 0)


# In[5]:


fig1 = plt.figure(figsize=(20,20))
mpl.rcParams['font.family']='SimHei'
plt.subplot(1,3,1)
plt.title('彩图')
plt.imshow(img_rgb)
plt.subplot(1,3,2)
plt.title('灰度图')
plt.imshow(img_gray,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(img_binary,cmap='gray')
plt.title('黑白二值图')
plt.show()
cv2.waitKey()

# In[ ]:




