import numpy as np
import cv2

def bilinear_interpolation(img,out_dim):
    srch,srcw,channel = img.shape
    dsth,dstw=out_dim[1],out_dim[0]
    if dsth == srch and dstw == dsth:
        return img.copy()
    dstimg=np.zeros((dsth,dstw,3),dtype=np.uint8)
    scalex,scaley = float(srcw)/dstw,float(srch)/dsth
    for i in range(3):
        for dsty in range(dsth):
            for dstx in range(dstw):

                srcx = (dstx+0.5)*scalex-0.5
                srcy = (dsty+0.5)*scaley-0.5
                srcx0= int(np.floor(srcx))
                srcx1= min(srcx0+1,srcw-1)
                srcy0= int(np.floor(srcy))
                srcy1= min(srcy0+1,srch-1)

                temp0 = (srcx1 - srcx) * img[srcy0, srcx0, i] + (srcx - srcx0) * img[srcy0, srcx1, i]
                temp1 = (srcx1 - srcx) * img[srcy1, srcx0, i] + (srcx - srcx0) * img[srcy1, srcx1, i]
                dstimg[dsty, dstx, i] = int((srcy1 - srcy) * temp0 + (srcy - srcy0) * temp1)
    return dstimg

img=cv2.imread("bat.jpg")
out_dim=[1000,1000]
bilimg=bilinear_interpolation(img,out_dim)
cv2.imshow("bilinear interpolation",bilimg)
cv2.waitKey(0)