
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


def bilinear_interpolation(img,out_dim):
    height,width,channels = img.shape
    scale_h,scale_w = height/out_dim[1],width/out_dim[0]
    emplyImage = np.zeros((out_dim[1],out_dim[0],channels),dtype=np.uint8)
    for i in range(out_dim[1]):
        for j in range(out_dim[0]):
            h,w = (i+0.5)*scale_h,(j+0.5)*scale_w
            h_org,w_org = h-0.5,w-0.5
            h_b0 = int(h_org)
            w_b0 = int(w_org)
            h_b1 = min(h_b0+1,height-1)
            w_b1 = min(w_b0+1,width-1)
            h_0_gray = (h_org-h_b0)*(img[h_b1,w_b0,:].astype(int)-img[h_b0,w_b0,:].astype(int))+img[h_b0,w_b0,:]
            h_1_gray = (h_org-h_b0)*(img[h_b1,w_b1,:].astype(int)-img[h_b0,w_b1,:].astype(int))+img[h_b0,w_b1,:]
            emplyImage[i,j,:] = ((w_org-w_b0)*(h_1_gray-h_0_gray)+h_0_gray).astype(int)
    return emplyImage


# In[3]:


img = cv2.imread('lenna.png')
img_new1 = bilinear_interpolation(img,(800,800))
cv2.imshow('bilinear interp',img_new1)
cv2.waitKey(0)


# In[4]:





# In[5]:





# In[6]:





# In[7]:





# In[8]:





# In[ ]:




