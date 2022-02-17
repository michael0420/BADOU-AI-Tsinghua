# -*- coding: utf-8 -*-
"""
@author:gwj

双线性差值
"""

import numpy as np
import cv2

def bilinear_interp(src,dst):
    srch,srcw,srcc=src.shape
    dsth,dstw,dstc=dst.shape
    scalew = srcw / dstw
    scaleh=srch/dsth
    if srch==dsth and srcw==dstw:
       return 0
    for k in range(dstc):
       for i in range(dsth):
          for j in range(dstw):
             srci = (i+0.5)*scaleh-0.5
             srcj = (j+0.5)*scalew-0.5
             u0=int(max(0,np.floor(srcj)))
             u1=int(min(u0+1,srcw-1))
             deltau=srcj-u0
             v0=int(max(0,np.floor(srci)))
             v1=int(min(v0+1,srch-1))
             deltav=srci-v0
             tempup=(1-deltau)*src[v0,u0,k]+deltau*src[v0,u1,k]
             tempdown=(1-deltau)*src[v1,u0,k]+deltau*src[v1,u1,k]
             dst[i,j,k]=int((1-deltav)*tempup+deltav*tempdown)

		   
if __name__=='__main__'	:
     img=cv2.imread("lenna.png")
     dsth=700
     dstw=700
     dst=np.zeros((dsth,dstw,3),img.dtype)
     bilinear_interp(img,dst)
     cv2.imshow("src",img)
     cv2.imshow("dst",dst)
     cv2.waitKey(0)
	 
	  

 