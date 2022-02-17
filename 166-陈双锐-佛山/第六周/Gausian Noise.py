import cv2
import random

def gausianNoise(img, mu, sigma, percent):
	h,w = img.shape
	noiseNum = int(h*w*percent)
	for i in range(noiseNum):
		x = random.randint(0,h-1)
		y = random.randint(0,w-1)
		img[x, y] = img[x, y] + random.gauss(mu, sigma)
		if img[x, y] > 255:
			img[x, y] = 255
		if img[x, y] < 0:
			img[x, y] = 0
	
	
	return img


img0 = cv2.imread("lenna.png", 0)
mu = 50
sigma = 1
img1 = gausianNoise(img0.copy(), mu, sigma, 0.8)
cv2.imshow("origin img", img0)
cv2.imshow("gauss noise", img1)
cv2.waitKey()



		
		
		
		
		
		
	
	
	
	