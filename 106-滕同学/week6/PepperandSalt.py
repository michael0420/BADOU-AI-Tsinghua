import numpy as np
import cv2
from numpy import shape
import random


def PepperandSalt(src, percetage):     
    NoiseImg = src    
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])    
    for i in range(NoiseNum): 
	    randX = random.randint(0, src.shape[0] - 1)       
	    randY = random.randint(0, src.shape[1] - 1) 
	    if random.random() <= 0.5:           
	    	NoiseImg[randX, randY] = 0       
	    else:            
	    	NoiseImg[randX, randY] = 255

    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = PepperandSalt(np.copy(img), 0.1)
cv2.imwrite('PepperandSalt.png', img1)
