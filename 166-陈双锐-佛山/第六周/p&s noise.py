import cv2
import random

def gausianNoise(img, percent):
	h,w = img.shape
	noiseNum = int(h*w*percent)
	for i in range(noiseNum):
		x = random.randint(0,h-1)
		y = random.randint(0,w-1)
		if random.random() < 0.5:
			img[x,y] = 0
		else:
			img[x,y] = 255
	
	return img


img0 = cv2.imread("lenna.png", 0)
img1 = gausianNoise(img0.copy(), 0.1)
cv2.imshow("origin img", img0)
cv2.imshow("p&s noise", img1)
cv2.waitKey()



		
		
		
		
		
		
	
	
	
	