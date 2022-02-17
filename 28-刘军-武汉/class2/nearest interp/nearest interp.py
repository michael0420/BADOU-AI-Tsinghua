
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


def nearest_interp(img):
    (height,width,channels)=img.shape
    emptyImage = np.zeros((800,800,channels),img.dtype)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            emptyImage[i,j,:] = img[x,y,:]
    return emptyImage


# In[3]:


img = cv2.imread('lenna.png')
new_img = nearest_interp(img)
cv2.imshow('image',img)
cv2.imshow('nearest interp',new_img)
cv2.waitKey(0)


# In[ ]:




